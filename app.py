"""
Plateforme de recherche des normes et fascicules techniques — LPEE
=====================================================================
Interface de type "chat" avec historique : les agents peuvent poser une
question, puis enchaîner des questions de suivi ("et pour le GNA ?") qui
tiennent compte de la conversation précédente.

À déployer sur Streamlit Community Cloud (gratuit), connecté à Supabase.

Secrets requis (Streamlit Cloud > Settings > Secrets, format .toml) :
    GOOGLE_API_KEY = "..."
    SUPABASE_URL = "..."
    SUPABASE_KEY = "..."   # doit autoriser la suppression (clé service_role ou policy DELETE)
    SUPABASE_TABLE = "documents"   # (optionnel) nom de la table des extraits
"""

import hashlib
import html as _html
import hmac
import io
import json
import os
import re
import threading
import time
import unicodedata

import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime, timedelta
from streamlit_cookies_controller import CookieController
from supabase import create_client
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from postgrest.exceptions import APIError  # noqa: F401  (erreurs Supabase lisibles)

st.set_page_config(page_title="Recherche Normes LPEE", page_icon="📚", layout="wide")

# ==========================================
# GESTION DES UTILISATEURS (stockés sur Hugging Face, mots de passe hachés)
# ==========================================
FICHIER_UTILISATEURS = "utilisateurs.json"
COMPTES_PAR_DEFAUT = {
    "admin": {"password": "admin123", "nom": "Administrateur LPEE", "role": "admin"},
    "agent1": {"password": "lpee2026", "nom": "Agent Laboratoire Béton", "role": "agent"},
    "agent2": {"password": "lpee2026", "nom": "Agent Laboratoire Sols", "role": "agent"},
}


def hacher_mdp(mdp: str, sel: str = None) -> str:
    sel = sel or os.urandom(16).hex()
    h = hashlib.pbkdf2_hmac("sha256", mdp.encode(), bytes.fromhex(sel), 200_000).hex()
    return f"pbkdf2${sel}${h}"


def verifier_mdp(mdp: str, stocke: str) -> bool:
    if stocke.startswith("pbkdf2$"):
        _, sel, _h = stocke.split("$")
        return hmac.compare_digest(hacher_mdp(mdp, sel), stocke)
    return hmac.compare_digest(mdp.encode(), stocke.encode())  # ancien format en clair


def charger_utilisateurs():
    """Lit la liste des comptes sur Hugging Face. Retourne None si la lecture échoue
    (pour ne jamais écraser les vrais comptes par les comptes par défaut)."""
    repo, token = st.secrets.get("HF_REPO_ID", ""), st.secrets.get("HF_TOKEN", "")
    defaut = {k: {**v, "password": hacher_mdp(v["password"])} for k, v in COMPTES_PAR_DEFAUT.items()}
    if not (repo and token):
        return defaut
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, RevisionNotFoundError
        try:
            chemin = hf_hub_download(repo_id=repo, filename=FICHIER_UTILISATEURS, repo_type="dataset",
                                     token=token, force_download=True)
        except (EntryNotFoundError, RevisionNotFoundError):
            return defaut  # première utilisation
        with open(chemin, encoding="utf-8") as f:
            return json.load(f) or defaut
    except Exception:
        return None


def sauver_utilisateurs(users: dict):
    repo, token = st.secrets.get("HF_REPO_ID", ""), st.secrets.get("HF_TOKEN", "")
    if not (repo and token):
        return False, "HF_REPO_ID / HF_TOKEN manquants."
    try:
        from huggingface_hub import HfApi
        HfApi(token=token).upload_file(
            path_or_fileobj=io.BytesIO(json.dumps(users, ensure_ascii=False, indent=1).encode("utf-8")),
            path_in_repo=FICHIER_UTILISATEURS, repo_id=repo, repo_type="dataset",
            commit_message="Mise à jour des comptes",
        )
        return True, ""
    except Exception as e:
        return False, str(e)[:300]


def modifier_comptes(fn):
    """Relit les comptes, applique fn(users) (retourne un message d'erreur ou None), puis sauvegarde."""
    users = charger_utilisateurs()
    if users is None:
        return False, "Lecture des comptes impossible, réessayez."
    err = fn(users)
    if err:
        return False, err
    ok, e = sauver_utilisateurs(users)
    if ok:
        st.session_state.utilisateurs = users
    return ok, e


# --- Connexion persistante (cookie signé) : reste connecté après un rafraîchissement ou un redéploiement ---
COOKIE_NOM = "lpee_auth"
DUREE_COOKIE_JOURS = 7


def _cle_signature() -> bytes:
    return (st.secrets.get("COOKIE_SECRET") or st.secrets.get("HF_TOKEN") or "lpee-secret").encode()


def creer_jeton(login: str, empreinte_mdp: str) -> str:
    expire = int(time.time()) + DUREE_COOKIE_JOURS * 86400
    base = f"{login}|{expire}"
    sig = hmac.new(_cle_signature(), f"{base}|{empreinte_mdp}".encode(), hashlib.sha256).hexdigest()
    return f"{base}|{sig}"


def lire_jeton(jeton: str, users: dict):
    """Retourne le login si le jeton est valide, non expiré et le mot de passe inchangé."""
    try:
        login, expire, sig = str(jeton).rsplit("|", 2)
        if int(expire) < time.time() or login not in users:
            return None
        attendu = hmac.new(_cle_signature(), f"{login}|{expire}|{users[login]['password']}".encode(),
                           hashlib.sha256).hexdigest()
        return login if hmac.compare_digest(sig, attendu) else None
    except Exception:
        return None


def lire_cookie(controller, nom: str):
    """Lit un cookie sans jamais planter : au 1er affichage, le composant n'a pas encore reçu les
    cookies du navigateur (sa liste interne vaut None => TypeError selon la version de la bibliothèque).
    On renvoie alors None ; dès que le navigateur répond, l'application se relance et le cookie est lu."""
    try:
        return controller.get(nom)
    except (TypeError, AttributeError, KeyError):
        return None


def poser_cookie(controller, login: str, empreinte_mdp: str):
    try:
        controller.set(
            COOKIE_NOM, creer_jeton(login, empreinte_mdp),
            expires=datetime.now() + timedelta(days=DUREE_COOKIE_JOURS),
            max_age=DUREE_COOKIE_JOURS * 86400, same_site="lax",
        )
    except (TypeError, AttributeError):
        pass  # connexion valide pour la session ; seul le « rester connecté » est ignoré


if "authentifie" not in st.session_state:
    st.session_state.authentifie = False
if "username_courant" not in st.session_state:
    st.session_state.username_courant = ""
if "nom_utilisateur" not in st.session_state:
    st.session_state.nom_utilisateur = ""
if "role_utilisateur" not in st.session_state:
    st.session_state.role_utilisateur = ""

controller = CookieController()

if not st.session_state.authentifie and not st.session_state.get("deconnexion_volontaire"):
    _jeton = lire_cookie(controller, COOKIE_NOM)
    if _jeton:
        _users = charger_utilisateurs()
        _login = lire_jeton(_jeton, _users) if _users else None
        if _login:
            st.session_state.utilisateurs = _users
            st.session_state.authentifie = True
            st.session_state.username_courant = _login
            st.session_state.nom_utilisateur = _users[_login]["nom"]
            st.session_state.role_utilisateur = _users[_login]["role"]
            st.rerun()

# ==========================================
# EFFETS « IA » (animation d'accueil, fond interactif, style néon) — désactivables : secret EFFETS_IA = "false"
# ==========================================
EFFETS_IA = str(st.secrets.get("EFFETS_IA", "true")).strip().lower() != "false"

JS_FOND_IA = r"""
(function(){
  if (window.__iaFond || window.innerWidth < 700) return;
  window.__iaFond = true;
  var c = document.createElement('canvas'); c.id = 'ia-fond';
  c.style.cssText = 'position:fixed;inset:0;width:100vw;height:100vh;pointer-events:none;z-index:1;opacity:.5';
  document.body.appendChild(c);
  var ctx = c.getContext('2d'), dpr = Math.min(window.devicePixelRatio || 1, 2), W, H, mouse = {x:-999, y:-999};
  function size(){ W = c.width = innerWidth*dpr; H = c.height = innerHeight*dpr; }
  size(); addEventListener('resize', size);
  addEventListener('mousemove', function(e){ mouse.x = e.clientX*dpr; mouse.y = e.clientY*dpr; });
  var N = 55, pts = [];
  for (var i=0;i<N;i++) pts.push({x:Math.random()*W, y:Math.random()*H, vx:(Math.random()-.5)*.45*dpr, vy:(Math.random()-.5)*.45*dpr});
  var reduit = matchMedia('(prefers-reduced-motion: reduce)').matches;
  function loop(){
    if (!document.hidden){
      ctx.clearRect(0,0,W,H);
      var d = 150*dpr;
      for (var i=0;i<N;i++){
        var p = pts[i];
        if (!reduit){ p.x += p.vx; p.y += p.vy; }
        if (p.x<0||p.x>W) p.vx*=-1; if (p.y<0||p.y>H) p.vy*=-1;
        var mx = p.x-mouse.x, my = p.y-mouse.y, md = Math.hypot(mx,my);
        if (md < 120*dpr && md > 0){ p.x += mx/md*1.4*dpr; p.y += my/md*1.4*dpr; }
        for (var j=i+1;j<N;j++){
          var q = pts[j], dist = Math.hypot(p.x-q.x, p.y-q.y);
          if (dist < d){ ctx.strokeStyle = 'rgba(99,102,241,'+(0.30*(1-dist/d))+')'; ctx.lineWidth = dpr;
            ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(q.x,q.y); ctx.stroke(); }
        }
        if (md < 200*dpr){ ctx.strokeStyle = 'rgba(6,182,212,'+(0.5*(1-md/(200*dpr)))+')'; ctx.lineWidth = dpr;
          ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(mouse.x,mouse.y); ctx.stroke(); }
        ctx.fillStyle = 'rgba(124,58,237,.75)'; ctx.beginPath(); ctx.arc(p.x,p.y,1.8*dpr,0,6.283); ctx.fill();
      }
    }
    requestAnimationFrame(loop);
  }
  loop();
})();
"""


def injecter_fond_ia():
    """Réseau de neurones interactif en fond de page (suit la souris). Injecté une seule fois dans la page."""
    if not EFFETS_IA:
        return
    code = json.dumps(JS_FOND_IA)
    components.html(
        "<script>try{var d=window.parent.document;if(!d.getElementById('ia-fond-script')){"
        "var s=d.createElement('script');s.id='ia-fond-script';s.textContent=" + code + ";d.head.appendChild(s);}}"
        "catch(e){}</script>", height=0)


def animation_accueil_ia(nom: str, nb: int | None):
    """Écran d'accueil « IA » (~4,5 s) : pluie binaire, réseau neuronal, anneaux HUD, titre glitch, étapes de démarrage."""
    nom_js = json.dumps(str(nom).upper())
    etapes = ["Connexion sécurisée établie",
              "Chargement du modèle de langage",
              "Activation de la base vectorielle",
              f"{nb:,} extraits normatifs prêts".replace(",", " ") if nb else "Normes et fascicules prêts"]
    html_anim = r"""
<style>
 *{box-sizing:border-box} body{margin:0}
 #sc{position:relative;width:100%;height:340px;border-radius:18px;overflow:hidden;color:#e8f0ff;
     font-family:'Source Sans Pro','Segoe UI',Arial,sans-serif;
     background:radial-gradient(circle at 25% 15%,#16358a 0%,#0a1633 52%,#04081a 100%);
     box-shadow:0 0 0 1px rgba(110,231,255,.35),0 0 40px rgba(99,102,241,.45) inset;animation:ent .5s ease-out}
 @keyframes ent{from{opacity:0;transform:scale(.97)}to{opacity:1;transform:none}}
 canvas{position:absolute;inset:0;width:100%;height:100%}
 #scan{position:absolute;left:0;right:0;height:90px;top:-90px;pointer-events:none;
       background:linear-gradient(180deg,transparent,rgba(110,231,255,.18),transparent);animation:sw 2.4s linear infinite}
 @keyframes sw{to{top:340px}}
 #cont{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;gap:42px;padding:0 24px}
 #hud{position:relative;width:132px;height:132px;flex:none}
 .ring{position:absolute;inset:0;border-radius:50%;border:2px dashed rgba(110,231,255,.8);animation:rot 6s linear infinite}
 .ring.r2{inset:14px;border:2px solid transparent;border-top-color:#a78bfa;border-bottom-color:#22d3ee;animation:rot 2.2s linear infinite reverse}
 .ring.r3{inset:28px;border:1px dotted rgba(255,255,255,.55);animation:rot 9s linear infinite}
 @keyframes rot{to{transform:rotate(360deg)}}
 #core{position:absolute;inset:42px;border-radius:50%;display:flex;align-items:center;justify-content:center;
       font-weight:800;font-size:22px;letter-spacing:1px;color:#fff;
       background:radial-gradient(circle,#6366f1,#1d4ed8 70%);animation:pl 1.4s ease-in-out infinite;
       box-shadow:0 0 25px #6366f1,0 0 60px rgba(34,211,238,.6)}
 @keyframes pl{50%{transform:scale(1.12);box-shadow:0 0 35px #a78bfa,0 0 85px rgba(34,211,238,.9)}}
 #txt{min-width:300px;max-width:430px}
 #k{font-size:12px;letter-spacing:4px;color:#6ee7ff;text-transform:uppercase}
 .gl{position:relative;font-size:34px;font-weight:800;line-height:1.1;margin:6px 0 14px;color:#fff;
     text-shadow:0 0 18px rgba(110,231,255,.6);word-break:break-word}
 .gl::before,.gl::after{content:attr(data-t);position:absolute;left:0;top:0;width:100%;overflow:hidden}
 .gl::before{color:#22d3ee;animation:g1 2.2s infinite steps(1);text-shadow:none}
 .gl::after{color:#f472b6;animation:g2 2.2s infinite steps(1);text-shadow:none}
 @keyframes g1{0%,86%,100%{clip-path:inset(0 0 100% 0);transform:none}88%{clip-path:inset(15% 0 55% 0);transform:translate(-4px,0)}94%{clip-path:inset(60% 0 10% 0);transform:translate(3px,0)}}
 @keyframes g2{0%,84%,100%{clip-path:inset(0 0 100% 0);transform:none}86%{clip-path:inset(55% 0 20% 0);transform:translate(4px,0)}92%{clip-path:inset(10% 0 70% 0);transform:translate(-3px,0)}}
 .st{font-family:Consolas,'Courier New',monospace;font-size:13px;color:#b6c8ff;opacity:0;transform:translateX(-10px);
     transition:all .35s;margin:3px 0}
 .st.on{opacity:1;transform:none}.st b{color:#4ade80;margin-right:8px}
 #bar{height:5px;border-radius:9px;background:rgba(255,255,255,.12);margin-top:12px;overflow:hidden}
 #bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,#22d3ee,#a78bfa,#f472b6);
        box-shadow:0 0 12px #22d3ee;transition:width .4s}
 @media(max-width:560px){#hud{display:none}.gl{font-size:24px}}
</style>
<div id="sc">
 <canvas id="cv"></canvas><div id="scan"></div>
 <div id="cont">
  <div id="hud"><div class="ring"></div><div class="ring r2"></div><div class="ring r3"></div><div id="core">IA</div></div>
  <div id="txt">
   <div id="k">Initialisation de l'assistant IA</div>
   <div class="gl" id="t" data-t=""></div>
   <div id="lst"></div>
   <div id="bar"><i id="pg"></i></div>
  </div>
 </div>
</div>
<script>
const NOM=__NOM__, ETAPES=__ETAPES__, cv=document.getElementById('cv'), ctx=cv.getContext('2d');
const dpr=Math.min(devicePixelRatio||1,2), reduit=matchMedia('(prefers-reduced-motion: reduce)').matches;
cv.width=cv.clientWidth*dpr; cv.height=cv.clientHeight*dpr;
const W=cv.width,H=cv.height, col=Math.floor(W/(14*dpr)), gouttes=Array.from({length:col},()=>Math.random()*H);
const pts=Array.from({length:42},()=>({x:Math.random()*W,y:Math.random()*H,vx:(Math.random()-.5)*dpr,vy:(Math.random()-.5)*dpr}));
function dessiner(){
  ctx.fillStyle='rgba(4,8,26,.16)'; ctx.fillRect(0,0,W,H);
  ctx.font=(12*dpr)+'px Consolas,monospace';
  for(let i=0;i<col;i++){
    ctx.fillStyle='rgba(34,211,238,.45)'; ctx.fillText(Math.random()>.5?'1':'0', i*14*dpr, gouttes[i]);
    gouttes[i]+= (reduit?0:14*dpr*.6); if(gouttes[i]>H && Math.random()>.97) gouttes[i]=0; else if(gouttes[i]>H+200) gouttes[i]=0;
  }
  const d=120*dpr;
  for(const p of pts){ if(!reduit){p.x+=p.vx;p.y+=p.vy;} if(p.x<0||p.x>W)p.vx*=-1; if(p.y<0||p.y>H)p.vy*=-1; }
  for(let i=0;i<pts.length;i++){ for(let j=i+1;j<pts.length;j++){
    const dist=Math.hypot(pts[i].x-pts[j].x,pts[i].y-pts[j].y);
    if(dist<d){ctx.strokeStyle='rgba(167,139,250,'+(0.5*(1-dist/d))+')';ctx.lineWidth=dpr;ctx.beginPath();
      ctx.moveTo(pts[i].x,pts[i].y);ctx.lineTo(pts[j].x,pts[j].y);ctx.stroke();}}
    ctx.fillStyle='#e0f2fe';ctx.beginPath();ctx.arc(pts[i].x,pts[i].y,2*dpr,0,6.283);ctx.fill(); }
  requestAnimationFrame(dessiner);
}
dessiner();
// titre « glitch » écrit lettre par lettre
const T=document.getElementById('t'), full='BIENVENUE, '+NOM; let i=0;
const iv=setInterval(()=>{ i++; T.textContent=full.slice(0,i); T.setAttribute('data-t',full.slice(0,i)); if(i>=full.length)clearInterval(iv); },55);
// étapes de démarrage + barre de progression
const L=document.getElementById('lst'), pg=document.getElementById('pg');
ETAPES.forEach((e,k)=>{ setTimeout(()=>{ const d=document.createElement('div'); d.className='st'; d.innerHTML='<b>✓</b>'+e;
  L.appendChild(d); requestAnimationFrame(()=>d.classList.add('on')); pg.style.width=((k+1)/ETAPES.length*100)+'%'; },700+k*650); });
</script>
""".replace("__NOM__", nom_js).replace("__ETAPES__", json.dumps(etapes))
    components.html(html_anim, height=350)


CSS_IA = """
<style>
@keyframes ia-degrade {0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes ia-pouls  {0%,100%{box-shadow:0 0 0 0 rgba(34,197,94,.65)}70%{box-shadow:0 0 0 9px rgba(34,197,94,0)}}
@keyframes ia-halo   {0%,100%{box-shadow:0 0 0 1px rgba(59,130,246,.35),0 0 10px rgba(59,130,246,.15)}
                      50%{box-shadow:0 0 0 1px rgba(139,92,246,.75),0 0 24px rgba(139,92,246,.4)}}
@keyframes ia-reflet {0%{background-position:-200% 0}100%{background-position:200% 0}}
@keyframes ia-entree {from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes ia-barre  {0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}

/* Titre principal : dégradé animé */
.stApp h1 {background:linear-gradient(90deg,#1d4ed8,#7c3aed,#06b6d4,#1d4ed8);background-size:300% 100%;
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;
  animation:ia-degrade 8s ease infinite;}

/* Pastille « IA active » */
.badge-ia {display:inline-flex;align-items:center;gap:8px;padding:4px 14px;border-radius:999px;font-size:.85rem;
  font-weight:600;color:#0f172a;background:linear-gradient(90deg,#e0f2fe,#ede9fe);border:1px solid rgba(124,58,237,.35);}
.badge-ia .point {width:9px;height:9px;border-radius:50%;background:#22c55e;animation:ia-pouls 1.8s infinite;}

/* Barre « l'IA travaille » en haut de page pendant tout calcul */
.stApp[data-teststate="running"]::before {content:"";position:fixed;top:0;left:0;right:0;height:3px;z-index:999999;
  background:linear-gradient(90deg,transparent,#22d3ee,#7c3aed,#f472b6,transparent);
  box-shadow:0 0 14px #7c3aed;animation:ia-barre 1.1s linear infinite;}

/* Zone de saisie : halo lumineux */
[data-testid="stChatInput"] {border-radius:14px;animation:ia-halo 3.5s ease-in-out infinite;}

/* Messages : apparition en glissé + halo néon pour l'assistant */
[data-testid="stChatMessage"] {animation:ia-entree .45s ease-out;border-radius:14px;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
  animation:ia-entree .45s ease-out, ia-halo 4s ease-in-out infinite;}
[data-testid="stChatMessageAvatarAssistant"] {box-shadow:0 0 14px rgba(124,58,237,.7);}

/* Boutons : lueur au survol */
.stButton>button, .stFormSubmitButton>button, .stDownloadButton>button {transition:all .25s;}
.stButton>button:hover, .stFormSubmitButton>button:hover, .stDownloadButton>button:hover {
  box-shadow:0 0 0 1px #7c3aed,0 0 18px rgba(124,58,237,.55);transform:translateY(-1px);}

/* Barre latérale : liseré néon */
[data-testid="stSidebar"] {border-right:1px solid rgba(124,58,237,.35);box-shadow:4px 0 22px rgba(99,102,241,.15);}

/* Indicateur « l'IA réfléchit » (toujours visible pendant la recherche et la rédaction) */
@keyframes ia-onde  {0%{transform:scale(.6);opacity:.9}100%{transform:scale(2.1);opacity:0}}
@keyframes ia-rail  {0%{transform:translateX(-100%)}100%{transform:translateX(300%)}}
@keyframes ia-points{0%,20%{content:"."}40%{content:".."}60%,100%{content:"..."}}
.ia-penser {display:flex;align-items:center;gap:16px;padding:10px 4px;}
.ia-penser .orbe {position:relative;flex:none;width:34px;height:34px;border-radius:50%;
  background:radial-gradient(circle at 30% 30%,#a78bfa,#4f46e5);box-shadow:0 0 16px rgba(99,102,241,.8);}
.ia-penser .orbe::before,.ia-penser .orbe::after {content:"";position:absolute;inset:0;border-radius:50%;
  border:2px solid #22d3ee;animation:ia-onde 1.8s ease-out infinite;}
.ia-penser .orbe::after {animation-delay:.9s;border-color:#a78bfa;}
.ia-penser .titre {font-weight:700;}
.ia-penser .titre::after {content:"...";animation:ia-points 1.4s steps(1) infinite;display:inline-block;width:1.2em;}
.ia-penser small {display:block;opacity:.7;margin:2px 0 7px;}
.ia-penser .rail {width:min(380px,60vw);height:5px;border-radius:9px;background:rgba(124,58,237,.15);overflow:hidden;}
.ia-penser .rail i {display:block;width:33%;height:100%;border-radius:9px;
  background:linear-gradient(90deg,#22d3ee,#7c3aed,#f472b6);box-shadow:0 0 10px #7c3aed;animation:ia-rail 1.3s linear infinite;}

/* ================= BARRE LATÉRALE « IA » ================= */
@keyframes ia-tourne  {to{transform:rotate(360deg)}}
@keyframes ia-balaye  {0%{top:-30%}100%{top:130%}}
@keyframes ia-lueur   {0%,100%{filter:drop-shadow(0 0 4px rgba(34,211,238,.35))}50%{filter:drop-shadow(0 0 14px rgba(124,58,237,.75))}}
@keyframes ia-glisse  {from{opacity:0;transform:translateX(-18px)}to{opacity:1;transform:none}}
@keyframes ia-aurore  {0%{background-position:0% 0%}50%{background-position:100% 100%}100%{background-position:0% 0%}}
@keyframes ia-barres  {0%,100%{height:4px}50%{height:16px}}
@keyframes ia-reflet2 {0%{transform:translateX(-120%) skewX(-20deg)}100%{transform:translateX(220%) skewX(-20deg)}}
@keyframes ia-tirets  {to{background-position:200% 0}}

/* Fond : grille « circuit » + aurore qui dérive lentement */
[data-testid="stSidebar"] > div:first-child {
  background-image:
    linear-gradient(rgba(99,102,241,.07) 1px,transparent 1px),
    linear-gradient(90deg,rgba(99,102,241,.07) 1px,transparent 1px),
    radial-gradient(circle at 20% 0%,rgba(34,211,238,.20),transparent 45%),
    radial-gradient(circle at 90% 100%,rgba(124,58,237,.20),transparent 50%);
  background-size:26px 26px,26px 26px,200% 200%,200% 200%;
  animation:ia-aurore 18s ease-in-out infinite;}

/* Entrée en cascade des éléments */
[data-testid="stSidebar"] [data-testid="stElementContainer"],
[data-testid="stSidebar"] [data-testid="stExpander"] {animation:ia-glisse .55s ease-out both;}
[data-testid="stSidebar"] [data-testid="stElementContainer"]:nth-child(2){animation-delay:.08s}
[data-testid="stSidebar"] [data-testid="stElementContainer"]:nth-child(3){animation-delay:.16s}
[data-testid="stSidebar"] [data-testid="stElementContainer"]:nth-child(4){animation-delay:.24s}
[data-testid="stSidebar"] [data-testid="stElementContainer"]:nth-child(5){animation-delay:.32s}
[data-testid="stSidebar"] [data-testid="stElementContainer"]:nth-child(6){animation-delay:.40s}
[data-testid="stSidebar"] [data-testid="stElementContainer"]:nth-child(n+7){animation-delay:.48s}

/* LOGO : cadre néon rotatif + balayage « scanner » + lueur pulsante */
[data-testid="stSidebar"] [data-testid="stImage"] {
  position:relative;overflow:hidden;border-radius:18px;padding:3px;isolation:isolate;
  animation:ia-lueur 3.6s ease-in-out infinite;}
[data-testid="stSidebar"] [data-testid="stImage"]::before {
  content:"";position:absolute;left:50%;top:50%;width:230%;aspect-ratio:1;z-index:-1;
  margin:-115% 0 0 -115%;
  background:conic-gradient(from 0deg,#22d3ee,#7c3aed,#f472b6,transparent 55%,#22d3ee);
  animation:ia-tourne 5s linear infinite;}
[data-testid="stSidebar"] [data-testid="stImage"] img {
  position:relative;z-index:1;display:block;border-radius:15px;background:#fff;}
[data-testid="stSidebar"] [data-testid="stImage"]::after {
  content:"";position:absolute;left:3px;right:3px;height:28%;z-index:2;pointer-events:none;border-radius:15px;
  background:linear-gradient(180deg,transparent,rgba(34,211,238,.38),rgba(124,58,237,.18),transparent);
  mix-blend-mode:multiply;animation:ia-balaye 3.2s ease-in-out infinite;}

/* CARTE UTILISATEUR */
.carte-ia {position:relative;display:flex;gap:12px;align-items:center;padding:12px 14px;margin:6px 0 4px;
  border-radius:16px;overflow:hidden;
  background:linear-gradient(135deg,rgba(14,165,233,.12),rgba(124,58,237,.14));
  border:1px solid rgba(124,58,237,.35);box-shadow:0 0 18px rgba(99,102,241,.18);}
.carte-ia::after {content:"";position:absolute;top:0;bottom:0;width:40%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.35),transparent);
  animation:ia-reflet2 4.5s ease-in-out infinite;pointer-events:none;}
.avatar-ia {position:relative;flex:none;width:46px;height:46px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-weight:800;color:#fff;font-size:1rem;
  background:radial-gradient(circle at 30% 25%,#6366f1,#1d4ed8);box-shadow:0 0 14px rgba(99,102,241,.8);}
.avatar-ia::before {content:"";position:absolute;inset:-5px;border-radius:50%;
  border:2px solid transparent;border-top-color:#22d3ee;border-right-color:#a78bfa;animation:ia-tourne 2.2s linear infinite;}
.avatar-ia::after {content:"";position:absolute;inset:-10px;border-radius:50%;
  border:1px dashed rgba(124,58,237,.55);animation:ia-tourne 9s linear infinite reverse;}
.infos-ia {min-width:0;flex:1}
.etat-ia {display:flex;align-items:center;gap:6px;font-size:.68rem;letter-spacing:.4px;text-transform:uppercase;opacity:.8}
.pt-ia {width:8px;height:8px;border-radius:50%;background:#22c55e;display:inline-block;animation:ia-pouls 1.8s infinite}
.nom-ia {font-weight:700;font-size:.98rem;line-height:1.2;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.eq-ia {display:flex;align-items:flex-end;gap:3px;height:18px;margin-top:5px}
.eq-ia b {display:block;width:3px;height:4px;border-radius:2px;background:linear-gradient(#22d3ee,#7c3aed);
  animation:ia-barres 1s ease-in-out infinite}
.eq-ia b:nth-child(2){animation-delay:.12s}.eq-ia b:nth-child(3){animation-delay:.27s}
.eq-ia b:nth-child(4){animation-delay:.05s}.eq-ia b:nth-child(5){animation-delay:.38s}
.eq-ia b:nth-child(6){animation-delay:.2s}.eq-ia b:nth-child(7){animation-delay:.31s}
.eq-ia em {font-style:normal;font-size:.68rem;margin-left:8px;opacity:.75;align-self:center;
  background:linear-gradient(90deg,#0ea5e9,#7c3aed);-webkit-background-clip:text;background-clip:text;
  -webkit-text-fill-color:transparent;font-weight:700}

/* Boutons et menus de la barre latérale : scan au survol + liseré néon */
[data-testid="stSidebar"] .stButton>button {position:relative;overflow:hidden;}
[data-testid="stSidebar"] .stButton>button::after {content:"";position:absolute;top:0;bottom:0;width:45%;left:0;
  transform:translateX(-120%) skewX(-20deg);background:linear-gradient(90deg,transparent,rgba(124,58,237,.28),transparent);}
[data-testid="stSidebar"] .stButton>button:hover::after {animation:ia-reflet2 .8s ease-out;}
[data-testid="stSidebar"] [data-testid="stExpander"] {border-radius:12px;transition:all .25s;}
[data-testid="stSidebar"] [data-testid="stExpander"]:hover {
  box-shadow:inset 3px 0 0 #7c3aed,0 0 16px rgba(99,102,241,.3);transform:translateX(2px);}

/* Séparateurs : trait lumineux animé */
[data-testid="stSidebar"] hr {border:none;height:2px;
  background:linear-gradient(90deg,transparent,#22d3ee,#7c3aed,#f472b6,transparent);
  background-size:200% 100%;animation:ia-tirets 3.5s linear infinite;}

@media (prefers-reduced-motion: reduce){
  [data-testid="stSidebar"] *,[data-testid="stSidebar"] *::before,[data-testid="stSidebar"] *::after,
  .carte-ia *,.carte-ia::after,[data-testid="stSidebar"] > div:first-child{animation:none !important;}
  .stApp h1,.badge-ia .point,[data-testid="stChatInput"],[data-testid="stChatMessage"],.ia-penser *,.ia-penser *::before,.ia-penser *::after,
  .stApp[data-teststate="running"]::before{animation:none !important;}
}
</style>
"""

# --- ÉCRAN DE CONNEXION ---
if not st.session_state.authentifie:
    if EFFETS_IA:
        st.markdown(CSS_IA, unsafe_allow_html=True)
        injecter_fond_ia()
    # Image de la page de connexion (même dossier que app.py ; tolère espaces/underscores et majuscules)
    _dossier = os.path.dirname(os.path.abspath(__file__))
    _image_garde = None
    for _nom in ("Page de garde Veille normative.jpg", "Page_de_garde_Veille_normative.jpg",
                 "Page de garde Veille normative.jpeg", "Page de garde Veille normative.png"):
        for _base in (_dossier, os.getcwd()):
            if os.path.exists(os.path.join(_base, _nom)):
                _image_garde = os.path.join(_base, _nom)
                break
        if _image_garde:
            break

    if _image_garde:
        col_image, col_form = st.columns([1.1, 1], gap="large", vertical_alignment="center")
        with col_image:
            st.image(_image_garde, use_container_width=True)
    else:
        col_form = st.container()

    with col_form:
        st.title("🔐 Connexion — Plateforme LPEE")
        st.markdown("Veuillez vous identifier pour accéder aux normes et fascicules techniques.")

        with st.form("form_login"):
            username_input = st.text_input("Nom d'utilisateur").strip().lower()
            password_input = st.text_input("Mot de passe", type="password")
            submit_login = st.form_submit_button("Se connecter")

            if submit_login:
                users = charger_utilisateurs()
                if users is None:
                    st.error("Service d'authentification momentanément indisponible. Réessayez dans un instant.")
                elif username_input in users and verifier_mdp(password_input, users[username_input]["password"]):
                    st.session_state.utilisateurs = users
                    st.session_state.authentifie = True
                    st.session_state.username_courant = username_input
                    st.session_state.nom_utilisateur = users[username_input]["nom"]
                    st.session_state.role_utilisateur = users[username_input]["role"]
                    st.session_state.pop("deconnexion_volontaire", None)
                    st.session_state.anim_ia = True  # animation d'accueil « IA » au premier affichage
                    poser_cookie(controller, username_input, users[username_input]["password"])
                    time.sleep(1)  # laisse le navigateur enregistrer le cookie
                    st.rerun()
                else:
                    st.error("Nom d'utilisateur ou mot de passe incorrect.")
    st.stop()


# ==========================================
# APPLICATION PRINCIPALE
# ==========================================

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"  # DOIT être le même modèle que dans ingest.py
MODELES_GENERATION = [
    "gemini-2.5-flash",          # rapide + précis (réflexion désactivée => réponse immédiate)
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]
MODELE_REFORMULATION = "gemini-flash-lite-latest"
NB_CANDIDATS = 40          # extraits récupérés dans la base
NB_MIN_RESULTATS = 8       # minimum gardé même si la pertinence est faible
NB_MAX_RESULTATS = 20      # maximum envoyé au modèle (prompt plus court = réponse plus rapide)
ECART_PERTINENCE = 0.08    # on écarte les extraits trop éloignés du meilleur score
NB_ECHANGES_CONTEXTE = 4

try:  # délai maximal par appel Gemini : évite une attente de plusieurs minutes si le service ne répond plus
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"], http_options=types.HttpOptions(timeout=120_000))
except Exception:
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
try:  # idem pour Supabase : une recherche qui n'aboutit pas en 30 s affiche une erreur au lieu de rester bloquée
    from supabase.lib.client_options import ClientOptions
    supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"],
                             options=ClientOptions(postgrest_client_timeout=30))
except Exception:
    supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


def _chemin_historique(username: str) -> str:
    return f"historiques/{re.sub(r'[^a-z0-9_-]', '_', username.lower())}.json"


def charger_historique(username: str):
    """Recharge les conversations de l'utilisateur depuis le dépôt Hugging Face (None si aucune / erreur)."""
    repo, token = st.secrets.get("HF_REPO_ID", ""), st.secrets.get("HF_TOKEN", "")
    if not (repo and token):
        st.session_state["erreur_historique"] = "HF_REPO_ID / HF_TOKEN manquants dans les Secrets."
        return None
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, RevisionNotFoundError
        try:
            chemin = hf_hub_download(
                repo_id=repo, filename=_chemin_historique(username), repo_type="dataset",
                token=token, force_download=True,
            )
        except (EntryNotFoundError, RevisionNotFoundError):
            return None  # première connexion : pas encore d'historique
        with open(chemin, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("sessions"):
            return data["sessions"], data.get("session_courante")
    except Exception as e:
        st.session_state["erreur_historique"] = str(e)
    return None


def sauvegarder_historique():
    """Enregistre les conversations de l'utilisateur connecté dans le dépôt Hugging Face."""
    username = st.session_state.get("username_courant")
    repo, token = st.secrets.get("HF_REPO_ID", ""), st.secrets.get("HF_TOKEN", "")
    if not (username and repo and token):
        return
    allege = {}
    for nom, msgs in st.session_state.sessions.items():
        allege[nom] = [
            {**m, "sources": [{**c, "content": c["content"][:700]} for c in m["sources"]]} if m.get("sources") else m
            for m in msgs
        ]
    contenu = json.dumps(
        {"sessions": allege, "session_courante": st.session_state.session_courante},
        ensure_ascii=False,
    ).encode("utf-8")
    try:
        from huggingface_hub import HfApi
        HfApi(token=token).upload_file(
            path_or_fileobj=io.BytesIO(contenu),
            path_in_repo=_chemin_historique(username),
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"Historique {username}",
        )
        st.session_state.pop("erreur_historique", None)
    except Exception as e:
        st.session_state["erreur_historique"] = str(e)


@st.cache_resource(show_spinner="Chargement du modèle de recherche (une seule fois)...")
def charger_modele():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texte(texte: str, prefixe: str):
    modele = charger_modele()
    vecteur = modele.encode([f"{prefixe}: {texte}"], normalize_embeddings=True, show_progress_bar=False)
    return vecteur[0].tolist()


def _erreur_temporaire(e) -> bool:
    msg = str(e)
    return any(m in msg for m in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "504", "INTERNAL", "DEADLINE"))


def _config(max_output_tokens, modele, temperature=0.1):
    """Température basse = réponses factuelles ; réflexion désactivée sur 2.5 = beaucoup plus rapide."""
    cfg = {"temperature": temperature}
    if max_output_tokens:
        cfg["max_output_tokens"] = max_output_tokens
    if "2.5" in modele:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    return types.GenerateContentConfig(**cfg)


def reformuler_question(question: str, historique: list) -> str:
    """Rend la question autonome. Sautée quand ce n'est pas nécessaire (gain de 1 à 3 s)."""
    if not historique:
        return question
    try:
        if _fichiers_cibles(question):
            return question  # la question nomme déjà un document : ne pas y mêler les sujets précédents
    except Exception:
        pass
    mots = question.split()
    debut = question.lower().lstrip().split(" ")[0]
    if len(mots) >= 8 and debut not in ("et", "pour", "aussi", "alors", "donc", "mais", "même"):
        return question  # question déjà complète

    derniers = historique[-NB_ECHANGES_CONTEXTE:]
    fil = "\n".join(f"{'Agent' if h['role'] == 'user' else 'Assistant'} : {h['content'][:600]}" for h in derniers)
    prompt = f"""Conversation technique entre un agent LPEE et un assistant documentaire :

{fil}

Nouvelle question : "{question}"

Reformule cette question en une question AUTONOME et COMPLÈTE (remplace "il", "ça", "et pour X"...) uniquement
si elle dépend de la conversation. Si elle a déjà son propre sujet (titre de document, norme, thème nouveau),
recopie-la TELLE QUELLE : n'y ajoute jamais un sujet de la conversation qu'elle ne mentionne pas.
Réponds UNIQUEMENT par la question reformulée, sans guillemets ni commentaire."""
    try:
        resp = client.models.generate_content(
            model=MODELE_REFORMULATION, contents=prompt, config=_config(120, MODELE_REFORMULATION, 0.0)
        )
        return (resp.text or "").strip() or question
    except Exception:
        return question


# ==========================================
# SYNCHRONISATION HUGGING FACE -> INDEX (suppression automatique des normes retirées)
# ==========================================
INTERVALLE_SYNCHRO = 3 * 60 * 60  # ajouts + suppressions vérifiés toutes les 3 h (et au démarrage) : économise CPU et base


@st.cache_data(ttl=60, show_spinner=False)
def fichiers_pdf_existants():
    """Liste des PDF présents dans le dépôt Hugging Face (rafraîchie toutes les 60 s). None si erreur."""
    try:
        from huggingface_hub import HfApi
        fichiers = HfApi(token=st.secrets.get("HF_TOKEN", "")).list_repo_files(
            st.secrets.get("HF_REPO_ID", ""), repo_type="dataset")
        return frozenset(_norm_chemin(f) for f in fichiers if f.lower().endswith(".pdf"))
    except Exception:
        return None


def _norm_chemin(chemin) -> str:
    """Même forme pour les chemins Windows (\\) et Hugging Face (/), accents normalisés."""
    return unicodedata.normalize("NFC", str(chemin or "")).replace("\\", "/")


def _sans_accents(texte: str) -> str:
    """Minuscules sans accents (pour comparer « chaussee » et « Chaussée »)."""
    t = unicodedata.normalize("NFD", str(texte or "").lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def _chemin_norm(chunk) -> str:
    return _norm_chemin(chunk["metadata"].get("chemin"))


@st.cache_data(ttl=600, show_spinner=False)
def layout_coherent() -> bool:
    """Vérifie (sur un échantillon de l'index) que les chemins de Supabase correspondent à ceux de
    Hugging Face. Si oui, on peut masquer sans risque les normes supprimées."""
    existants = fichiers_pdf_existants()
    if not existants:
        return False
    table = st.secrets.get("SUPABASE_TABLE", "documents")
    chemins = []
    for debut in (0, 3000, 10000):
        try:
            r = supabase.table(table).select("chemin:metadata->>chemin").range(debut, debut + 59).execute()
            chemins += [l["chemin"] for l in (r.data or []) if l.get("chemin")]
        except Exception:
            continue
    if not chemins:
        return False
    return sum(1 for c in chemins if _norm_chemin(c) in existants) / len(chemins) >= 0.5


def chemins_indexes(table: str) -> set:
    """Chemins distincts présents dans l'index Supabase.
    Utilise la fonction SQL chemins_indexes() si elle existe (rapide), sinon parcourt la table."""
    try:
        r = supabase.rpc("chemins_indexes", {}).execute()
        return {l["chemin"] for l in (r.data or []) if l.get("chemin")}
    except Exception as e:
        msg = str(e)
        if "PGRST202" not in msg and "Could not find the function" not in msg:
            raise RuntimeError(f"Fonction SQL chemins_indexes() en erreur : {msg[:250]}")  # vraie cause visible
    chemins, debut = set(), 0
    while True:
        r = supabase.table(table).select("chemin:metadata->>chemin").range(debut, debut + 999).execute()
        lot = r.data or []
        chemins.update(l["chemin"] for l in lot if l.get("chemin"))
        if len(lot) < 1000:
            break
        debut += 1000
    return chemins


def synchroniser_index(existants, table: str, chemins_db=None) -> dict:
    """Supprime de Supabase les extraits des PDF qui n'existent plus sur Hugging Face.
    'coherent' = les chemins de l'index correspondent bien à ceux du dépôt (sinon le filtrage reste désactivé)."""
    if not existants:
        return {"supprimes": [], "echecs": [], "coherent": False,
                "erreur": "Liste des PDF Hugging Face indisponible ou vide : rien supprimé."}
    try:
        chemins_db = chemins_db if chemins_db is not None else chemins_indexes(table)
        orphelins = [c for c in chemins_db if _norm_chemin(c) not in existants]
        if orphelins and len(orphelins) > max(3, len(chemins_db) // 2):
            return {"supprimes": [], "echecs": [], "coherent": False, "erreur":
                    f"Sécurité : {len(orphelins)} documents sur {len(chemins_db)} de l'index ne correspondent à aucun "
                    "fichier du dépôt Hugging Face (arborescence différente ?). Rien n'est supprimé ni filtré."}

        supprimes, echecs = [], []
        for c in orphelins:
            r = supabase.table(table).delete().eq("metadata->>chemin", c).execute()
            (supprimes if r.data else echecs).append(c)
        return {"supprimes": supprimes, "echecs": echecs, "coherent": True, "erreur": None}
    except Exception as e:
        return {"supprimes": [], "echecs": [], "coherent": None, "erreur": str(e)[:300]}


# ==========================================
# INDEXATION AUTOMATIQUE DES NOUVELLES NORMES (Hugging Face -> Supabase)
# ==========================================
# ⚠️ À aligner sur ingest.py (même découpage, même préfixe "passage: ", même modèle d'embedding).
TAILLE_CHUNK = 1200       # identique à CHUNK_SIZE d'ingest.py
CHEVAUCHEMENT = 150       # identique à CHUNK_OVERLAP d'ingest.py
SEUIL_TEXTE = 20          # moins de caractères sur une page => page scannée => OCR (même seuil qu'ingest.py)
TAILLE_LOT = 64


# --- Nettoyage des « remplissages » (lignes de tirets, points de conduite des sommaires, bordures de tableaux) ---
CARS_REMPLISSAGE = set("-_.=*·…•─━—–~")
_RE_REPETITION = re.compile(r"([\-_.=*·…•─━—–~])\1{3,}")
_RE_ESPACES = re.compile(r"(?:[.\-_·]\s){5,}[.\-_·]?")


def nettoyer_remplissage(texte: str) -> str:
    """« ................ » ou « ---------- » -> « ... » / « --- » ; « . . . . . . » -> « … »."""
    if not texte:
        return texte
    texte = _RE_REPETITION.sub(lambda m: m.group(1) * 3, texte)
    texte = _RE_ESPACES.sub(" … ", texte)
    return re.sub(r"(?:[ \t]*(?:\.\.\.|…|---)[ \t]*\n){3,}", "…\n", texte)  # lignes entières de remplissage


def filtrer_remplissage_flux(flux, max_rep: int = 3):
    """Filtre le flux de la réponse : au plus 3 caractères de remplissage identiques d'affilée, et plus de
    2 lignes de suite faites uniquement de remplissage (le modèle recopie parfois les tirets des tableaux)."""
    dernier, rep = "", 0
    ligne_len, ligne_remplie, nb_lignes = 0, True, 0
    for morceau in flux:
        sortie = []
        for ch in morceau:
            if ch == "\n":
                if ligne_len > 0:
                    nb_lignes = nb_lignes + 1 if ligne_remplie else 0
                ignorer = ligne_len > 0 and ligne_remplie and nb_lignes > 2
                ligne_len, ligne_remplie, dernier, rep = 0, True, "", 0
                if not ignorer:
                    sortie.append(ch)
                continue
            ligne_len += 1
            if ch not in CARS_REMPLISSAGE and ch not in " |:\t":
                ligne_remplie = False
            elif nb_lignes >= 2 and ligne_remplie:
                continue  # 3e ligne (ou plus) de remplissage d'affilée : on l'ignore
            if ch in CARS_REMPLISSAGE and ch == dernier:
                rep += 1
                if rep > max_rep:
                    continue
            else:
                dernier, rep = ch, 1
            sortie.append(ch)
        if sortie:
            yield "".join(sortie)


def _texte_page(page) -> str:
    """Texte d'une page PDF : couche texte si elle existe, sinon OCR Tesseract (français)."""
    texte = page.get_text("text").strip()
    if len(texte) >= SEUIL_TEXTE:
        return nettoyer_remplissage(texte)
    try:
        import pytesseract
        from PIL import Image
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return nettoyer_remplissage(pytesseract.image_to_string(img, lang="fra").strip())
    except Exception:
        return nettoyer_remplissage(texte)


def _decouper(texte: str, taille=TAILLE_CHUNK, chevauchement=CHEVAUCHEMENT) -> list:
    """Même découpage que decouper_en_chunks() d'ingest.py."""
    chunks, debut, n = [], 0, len(texte)
    while debut < n:
        fin = min(debut + taille, n)
        if fin < n:
            dernier_espace = texte.rfind(" ", debut, fin)
            if dernier_espace > debut:
                fin = dernier_espace
        morceau = texte[debut:fin].strip()
        if morceau:
            chunks.append(morceau)
        debut = fin - chevauchement if fin - chevauchement > debut else fin
    return chunks


def indexer_pdf(chemin: str, table: str, modele, repo: str, token: str, suivi=None) -> int:
    """Télécharge un PDF du dépôt, extrait le texte (OCR si besoin) et l'ajoute à Supabase,
    avec exactement le même format de lignes qu'ingest.py."""
    import fitz  # PyMuPDF
    from huggingface_hub import hf_hub_download
    nom = chemin.split("/")[-1]
    if suivi is not None:
        suivi["progression"] = f"{nom} — téléchargement depuis Hugging Face"
    local = hf_hub_download(repo_id=repo, filename=chemin, repo_type="dataset", token=token)
    lignes = []
    with fitz.open(local) as doc:
        for num, page in enumerate(doc, 1):
            if suivi is not None:
                suivi["progression"] = f"{nom} — lecture/OCR page {num}/{len(doc)}"
            texte = _texte_page(page)
            if len(texte) < 20:
                continue
            for idx, morceau in enumerate(_decouper(texte)):
                cid = f"{num}-{idx}"
                lignes.append({
                    "content": morceau, "fichier": chemin, "chunk_id": cid,  # clé unique = chemin complet (deux PDF de même nom ne s'écrasent plus)
                    "metadata": {"fichier": nom, "chemin": chemin, "page": num, "chunk_id": cid},
                })
    if not lignes:
        raise ValueError("aucun texte extrait (OCR indisponible ? vérifiez packages.txt)")
    if suivi is not None:
        suivi["progression"] = f"{nom} — calcul des embeddings et envoi ({len(lignes)} extraits)"
    supabase.table(table).delete().eq("metadata->>chemin", chemin).execute()  # repart d'une base propre
    try:
        for i in range(0, len(lignes), TAILLE_LOT):
            lot = lignes[i:i + TAILLE_LOT]
            vecteurs = modele.encode([f"passage: {l['content']}" for l in lot],
                                     normalize_embeddings=True, show_progress_bar=False)
            for l, v in zip(lot, vecteurs):
                l["embedding"] = v.tolist()
            supabase.table(table).upsert(lot, on_conflict="fichier,chunk_id").execute()
    except Exception:
        try:
            supabase.table(table).delete().eq("metadata->>chemin", chemin).execute()  # pas d'indexation à moitié
        except Exception:
            pass
        raise
    return len(lignes)


SEUIL_BASE_MO = 430            # au-delà, l'indexation s'arrête (limite gratuite Supabase : 500 Mo)
MAX_FICHIERS_PAR_PASSE = 5     # indexation automatique : au plus 5 PDF par passage (le CPU Streamlit Cloud est limité)
MAX_FICHIERS_MANUEL = 8        # bouton « Synchroniser maintenant » : au plus 8 PDF par clic (secret MAX_FICHIERS_SYNCHRO pour changer)


def _max_manuel() -> int:
    try:
        return max(1, int(st.secrets.get("MAX_FICHIERS_SYNCHRO", MAX_FICHIERS_MANUEL)))
    except Exception:
        return MAX_FICHIERS_MANUEL


FICHIER_REGLAGES = "reglages_indexation.json"   # stocké dans le dépôt Hugging Face, comme utilisateurs.json
_LIRE = object()


def _lire_reglages():
    """Réglages d'indexation enregistrés sur Hugging Face : {"tous": bool, "dossiers": [...]}.
    None = aucun réglage enregistré (=> tous les dossiers). En cas d'erreur de lecture : on n'indexe RIEN (sécurité)."""
    repo, token = st.secrets.get("HF_REPO_ID", ""), st.secrets.get("HF_TOKEN", "")
    if not (repo and token):
        return None
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, RevisionNotFoundError
        try:
            chemin = hf_hub_download(repo_id=repo, filename=FICHIER_REGLAGES, repo_type="dataset",
                                     token=token, force_download=True)
        except (EntryNotFoundError, RevisionNotFoundError):
            return None
        with open(chemin, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"tous": False, "dossiers": [], "erreur": True}


@st.cache_data(ttl=30, show_spinner=False)
def charger_reglages():
    return _lire_reglages()


def sauver_reglages(reglages: dict):
    repo, token = st.secrets.get("HF_REPO_ID", ""), st.secrets.get("HF_TOKEN", "")
    if not (repo and token):
        return False, "HF_REPO_ID / HF_TOKEN manquants."
    try:
        from huggingface_hub import HfApi
        HfApi(token=token).upload_file(
            path_or_fileobj=io.BytesIO(json.dumps(reglages, ensure_ascii=False, indent=1).encode("utf-8")),
            path_in_repo=FICHIER_REGLAGES, repo_id=repo, repo_type="dataset",
            commit_message="Mise à jour des dossiers à indexer",
        )
        charger_reglages.clear()
        return True, ""
    except Exception as e:
        return False, str(e)[:300]


def _secret_dossiers():
    """Secret DOSSIERS_INDEXES (liste TOML ou texte séparé par « ; »). "*" = tous. Prioritaire sur les réglages."""
    try:
        brut = st.secrets.get("DOSSIERS_INDEXES", "")
    except Exception:
        return None
    liste = [str(x) for x in brut] if isinstance(brut, (list, tuple)) else str(brut).replace("|", ";").split(";")
    liste = [x.strip() for x in liste if x.strip()]
    return liste or None


def _dossiers_autorises(reg=_LIRE):
    """None = tous les dossiers ; sinon liste (éventuellement vide) des dossiers que l'application peut indexer."""
    liste = _secret_dossiers()
    if liste:
        return None if "*" in liste else [_norm_chemin(x).strip("/").lower() for x in liste]
    if reg is _LIRE:
        reg = _lire_reglages()
    if reg is None or reg.get("tous"):
        return None
    return [_norm_chemin(x).strip("/").lower() for x in reg.get("dossiers", [])]


def _autorise(chemin, autorises) -> bool:
    return autorises is None or any(_norm_chemin(chemin).lower().startswith(d + "/") for d in autorises)


def _dossier_racine(chemin):
    c = _norm_chemin(chemin)
    return c.split("/")[0] if "/" in c else None


@st.cache_data(ttl=120, show_spinner=False)
def stats_dossiers_indexes(table: str) -> dict:
    """Nombre de PDF indexés par dossier de premier niveau."""
    try:
        chemins = chemins_indexes(table)
    except Exception:
        return {}
    d = {}
    for c in chemins:
        k = _dossier_racine(c)
        if k:
            d[k] = d.get(k, 0) + 1
    return d


def _taille_base_directe():
    try:
        return int(supabase.rpc("taille_base", {}).execute().data) / 1048576
    except Exception:
        return None


def indexer_nouveaux(existants, table: str, modele, repo: str, token: str, chemins_db=None, suivi=None,
                     maximum=None) -> dict:
    """Indexe les PDF présents sur Hugging Face mais absents de l'index Supabase."""
    res = {"indexes": [], "echecs": {}, "erreur": None, "restants": 0}
    if not existants:
        return res
    try:
        deja = {_norm_chemin(c) for c in (chemins_db if chemins_db is not None else chemins_indexes(table))}
    except Exception as e:
        res["erreur"] = str(e)[:300]
        return res
    a_faire = sorted(set(existants) - deja)
    autorises = _dossiers_autorises()
    if autorises is not None:
        a_faire = [c for c in a_faire if _autorise(c, autorises)]
    connus = (suivi or {}).get("echecs_connus", {})
    a_faire.sort(key=lambda c: c in connus)  # les fichiers déjà en échec passent en dernier : ils ne bloquent pas la file
    total_a_faire = len(a_faire)
    if maximum:
        a_faire = a_faire[:maximum]
    traites = 0
    for chemin in a_faire:
        mo = _taille_base_directe()
        if mo is not None and mo >= SEUIL_BASE_MO:
            res["erreur"] = (f"Indexation suspendue : la base fait {mo:.0f} Mo (seuil {SEUIL_BASE_MO} Mo sur 500). "
                             "Libérez de l'espace ou passez à l'offre Pro.")
            break
        traites += 1
        try:
            n = indexer_pdf(chemin, table, modele, repo, token, suivi)
            res["indexes"].append(f"{chemin} ({n} extraits)")
            if suivi is not None:
                suivi.setdefault("echecs_connus", {}).pop(chemin, None)
        except Exception as e:
            res["echecs"][chemin] = str(e)[:200]
            if suivi is not None:
                suivi.setdefault("echecs_connus", {})[chemin] = str(e)[:200]
    res["restants"] = max(0, total_a_faire - len(res["indexes"]) - len(res["echecs"]))
    if suivi is not None:
        suivi["progression"] = None
    if res["indexes"]:
        _candidats.clear()  # les recherches mises en cache (1 h) ne verraient pas la nouvelle norme
    return res


@st.cache_resource
def _etat_synchro():
    # filtre_sur : True seulement après avoir vérifié que l'index et Hugging Face utilisent les mêmes chemins
    return {"dernier": 0.0, "verrou": threading.Lock(), "resultat": None, "filtre_sur": False}


def _appliquer_resultat(etat, r):
    etat["resultat"] = r
    if r.get("coherent") is not None:
        etat["filtre_sur"] = bool(r["coherent"])


def lancer_synchro_auto():
    """Synchronisation en arrière-plan (ajouts + suppressions), sans ralentir l'utilisateur."""
    etat = _etat_synchro()
    if time.time() - etat["dernier"] < INTERVALLE_SYNCHRO:
        return
    existants = fichiers_pdf_existants()
    if not existants:
        return
    modele = charger_modele()  # chargé ici (thread principal), pas dans le thread d'arrière-plan
    if not etat["verrou"].acquire(blocking=False):
        return
    etat["dernier"] = time.time()
    table = st.secrets.get("SUPABASE_TABLE", "documents")
    repo, token = st.secrets.get("HF_REPO_ID", ""), st.secrets.get("HF_TOKEN", "")
    # INDEXATION_AUTO = "false" dans les secrets : pas d'indexation automatique (utile pendant une
    # indexation massive avec ingest.py sur votre PC, pour éviter que les deux travaillent en même temps)
    auto = str(st.secrets.get("INDEXATION_AUTO", "true")).strip().lower() not in ("false", "0", "non", "no")

    def _tache():
        try:
            etat["debut"] = time.time()
            etat["progression"] = "lecture de l'index Supabase"
            try:
                en_base = chemins_indexes(table)  # une seule lecture de l'index pour les deux étapes
            except Exception:
                en_base = None
            etat["progression"] = "nettoyage des normes supprimées"
            _appliquer_resultat(etat, synchroniser_index(existants, table, en_base))
            if auto:
                etat["indexation"] = indexer_nouveaux(existants, table, modele, repo, token, en_base, etat,
                                                      maximum=MAX_FICHIERS_PAR_PASSE)
            else:
                etat["indexation"] = {"indexes": [], "echecs": {}, "erreur":
                                      "Indexation automatique désactivée (secret INDEXATION_AUTO = false)."}
        finally:
            etat["progression"] = None
            etat["verrou"].release()

    threading.Thread(target=_tache, daemon=True).start()


# --- Normes principales : dossier "normes principales" du dépôt Hugging Face ---
BONUS_PRINCIPALE = 0.04          # avantage de classement donné aux normes principales
NB_CANDIDATS_PRINCIPALES = 25    # extraits cherchés spécifiquement dans ce dossier
MOTIF_SQL_PRINCIPALE = "normes_principal%/%"  # '_' = joker : accepte "normes principales" et "normes_principales"


def est_principale(chunk) -> bool:
    """Vrai si le document est dans le dossier « normes principales » (accents/majuscules/_ ignorés)."""
    chemin = _chemin_norm(chunk)
    if "/" not in chemin:
        return False
    premier = unicodedata.normalize("NFD", chemin.split("/")[0])
    premier = "".join(ch for ch in premier if unicodedata.category(ch) != "Mn")
    return premier.lower().replace("_", " ").replace("-", " ").strip().startswith("normes principal")


def _rpc(nom: str, params: dict, essais: int = 2):
    """Appel RPC Supabase : une 2e tentative en cas de délai dépassé / saturation,
    sinon une erreur LISIBLE (Streamlit Cloud masque le message d'origine de APIError)."""
    derniere = None
    for tentative in range(essais):
        try:
            return supabase.rpc(nom, params).execute()
        except Exception as e:
            derniere = e
            code = str(getattr(e, "code", "") or "")
            msg = str(getattr(e, "message", "") or e)
            temporaire = code in ("57014", "53300", "54000", "PGRST003") or "timeout" in msg.lower()
            if tentative + 1 < essais and temporaire:
                time.sleep(1.5)
                continue
            break
    code = str(getattr(derniere, "code", "") or "")
    details = " | ".join(str(x) for x in (getattr(derniere, "message", None) or derniere,
                                          getattr(derniere, "details", None),
                                          getattr(derniere, "hint", None)) if x)
    raise RuntimeError(f"Supabase · fonction « {nom} » · code {code or 'n/a'} · {details}"[:700])


@st.cache_data(ttl=60, show_spinner=False)
def nb_extraits():
    """Nombre de lignes dans la table des extraits, None si indisponible."""
    try:
        table = st.secrets.get("SUPABASE_TABLE", "documents")
        return supabase.table(table).select("id", count="exact").limit(1).execute().count
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def taille_base_mo():
    """Taille de la base en Mo (fonction SQL taille_base()), None si indisponible."""
    try:
        return int(_rpc("taille_base", {}, essais=1).data) / 1048576
    except Exception:
        return None


@st.cache_data(ttl=3600, max_entries=300, show_spinner=False)
def _vecteur_question(question: str):
    return embed_texte(question, prefixe="query")


@st.cache_data(ttl=3600, max_entries=300, show_spinner=False)
def _candidats(question: str):
    try:
        res = _rpc("match_documents", {
            "query_embedding": _vecteur_question(question),
            "match_count": NB_CANDIDATS,
        })
    except RuntimeError as e:
        if "57014" not in str(e):
            raise
        time.sleep(2)  # délai Supabase dépassé : 2e essai plus léger (moins de candidats = recherche plus courte)
        res = _rpc("match_documents", {
            "query_embedding": _vecteur_question(question),
            "match_count": max(10, NB_CANDIDATS // 3),
        })
    if not res.data:
        # Une exception n'est PAS mise en cache par Streamlit (contrairement à une liste vide gardée 1 h)
        raise LookupError("La fonction match_documents n'a renvoyé aucun extrait : la table est vide, "
                          "ou l'index/la fonction SQL ne trouve rien (voir schema.sql).")
    return sorted(res.data, key=lambda c: c["similarity"], reverse=True)


@st.cache_data(ttl=3600, max_entries=300, show_spinner=False)
def _candidats_principales(question: str):
    res = _rpc("match_documents_prefixe", {
        "query_embedding": _vecteur_question(question),
        "match_count": NB_CANDIDATS_PRINCIPALES,
        "motif": MOTIF_SQL_PRINCIPALE,
    })
    return sorted(res.data or [], key=lambda c: c["similarity"], reverse=True)


def _principales(question: str, generaux: list) -> list:
    """Extraits des normes principales. Si la fonction SQL dédiée n'existe pas encore,
    on se rabat sur ceux déjà présents parmi les candidats généraux."""
    try:
        return _candidats_principales(question)
    except Exception:
        return [c for c in generaux if est_principale(c)]


# --- Recherche ciblée par numéro de norme (ex. « NM EN 12350-2 ») ---
# La recherche vectorielle seule est mauvaise pour les identifiants : le nom du fichier n'est pas dans le texte
# embeddé, donc « 12350-2 » ressemble à « 12504-2 » ou « 934-2 ». On repère donc le numéro dans la question,
# on retrouve le(s) PDF correspondant(s) par leur nom, et on prend leurs extraits les plus proches de la question.
BONUS_CIBLE = 0.10
MAX_FICHIERS_CIBLES = 4
NB_EXTRAITS_CIBLES = 8
NB_CIBLES_GARANTIS = 10  # extraits du document cité toujours transmis au modèle


def _nettoyer_ref(texte: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", _sans_accents(texte)).strip() + " "


def _refs_dans_question(question: str) -> list:
    """[(numero, partie|None)] : « NM EN 12350-2 » -> [('12350', '2')]. Ignore les années isolées (2021...)."""
    refs = []
    for m in re.finditer(r"(?<!\d)(\d{3,5})(?:\s*[-\u2013\u2011/]\s*(\d{1,2}))?(?!\d)", question):
        num, partie = m.group(1), m.group(2)
        if partie is None and len(num) == 4 and 1900 <= int(num) <= 2100:
            continue
        refs.append((num, partie))
    return refs


MOTS_VIDES = {"de", "des", "du", "la", "le", "les", "un", "une", "et", "en", "pour", "sur", "au", "aux", "dans",
              "par", "avec", "sans", "ou", "ce", "ces", "qui", "que", "quel", "quelle", "quels", "quelles", "est",
              "sont", "pdf", "the", "of", "and"}


def _mots_utiles(texte: str) -> set:
    """Mots significatifs (sans accents, sans mots vides, pluriels simplifiés) : « chaussées » -> « chausse »."""
    mots = set()
    for m in _nettoyer_ref(texte).split():
        if m in MOTS_VIDES or (len(m) <= 2 and not m.isdigit()):
            continue
        if len(m) > 3 and m.endswith(("s", "x")):
            m = m[:-1]
        mots.add(m)
    return mots


def _fichiers_cibles(question: str) -> list:
    """PDF visés par la question : par numéro de norme (« 12350-2 ») ou par titre (« Catalogue des structures types »)."""
    existants = fichiers_pdf_existants()
    if not existants:
        return []
    refs = _refs_dans_question(question)
    mots_q = _mots_utiles(question)
    notes = {}
    for chemin in sorted(existants):
        base = chemin.split("/")[-1].rsplit(".", 1)[0]
        nom = _nettoyer_ref(base)
        # 1) numéro de norme présent dans le nom du fichier
        for num, partie in refs:
            motif = f" {num} {partie} " if partie else f" {num} "
            if motif in nom:
                notes[chemin] = max(notes.get(chemin, 0), 2.0)
        # 2) titre : presque tous les mots du nom du fichier figurent dans la question
        mots_f = _mots_utiles(base)
        if len(mots_f) >= 2 and mots_q:
            couverture = len(mots_f & mots_q) / len(mots_f)
            if couverture >= 0.8:
                notes[chemin] = max(notes.get(chemin, 0), 1.0 + couverture)
        # 3) la question reprend un morceau précis du nom (au moins 3 mots, tous présents dans le nom)
        if len(mots_q) >= 3 and mots_q <= mots_f:
            notes[chemin] = max(notes.get(chemin, 0), 1.5)
    classes = sorted(notes, key=lambda c: (-notes[c], c))
    return classes[:MAX_FICHIERS_CIBLES]


def _diversifier(lignes: list, max_par_page: int = 2, n: int = NB_EXTRAITS_CIBLES) -> list:
    """Évite que tous les extraits viennent de la même page (ex. page de garde en double)."""
    comptes, sortie = {}, []
    for l in lignes:
        page = l["metadata"].get("page")
        if comptes.get(page, 0) >= max_par_page:
            continue
        comptes[page] = comptes.get(page, 0) + 1
        sortie.append(l)
        if len(sortie) >= n:
            break
    return sortie


def _apercu_document(chemin: str, table: str, n: int = 10) -> list:
    """Vue d'ensemble d'un document : sommaire éventuel, début, puis extraits répartis sur toutes les pages."""
    r = (supabase.table(table).select("id,content,metadata").eq("fichier", chemin)
         .order("chunk_id").limit(2000).execute())
    lignes = r.data or []
    if not lignes:
        return []
    choisis = [l for l in lignes if "sommaire" in (l["content"] or "").lower()
               or "table des mati" in (l["content"] or "").lower()][:2]
    choisis += lignes[:2]
    reste = max(1, n - len(choisis))
    pas = max(1, len(lignes) // reste)
    choisis += lignes[pas // 2::pas][:reste]
    vus, sortie = set(), []
    for l in choisis:
        if l["id"] in vus:
            continue
        vus.add(l["id"])
        l["similarity"] = 0.80
        sortie.append(l)
    return sortie[:n]


def _candidats_cibles(question: str) -> list:
    fichiers = _fichiers_cibles(question)
    st.session_state["diag_cibles"] = {}
    if not fichiers:
        return []
    table = st.secrets.get("SUPABASE_TABLE", "documents")
    mots_q = _mots_utiles(question)
    sortie = []
    for chemin in fichiers:
        lignes = []
        # La question ne fait que nommer le document -> on en donne un aperçu (objet, plan, contenu)
        titre_seul = len(mots_q) >= 2 and mots_q <= _mots_utiles(chemin.split("/")[-1].rsplit(".", 1)[0])
        if titre_seul:
            try:
                lignes = _apercu_document(chemin, table)
            except Exception:
                lignes = []
        if not lignes:
            try:  # extraits du fichier les plus proches de la question
                res = _rpc("match_documents_prefixe", {
                    "query_embedding": _vecteur_question(question),
                    "match_count": NB_EXTRAITS_CIBLES * 2,
                    "motif": chemin,
                })
                lignes = _diversifier(res.data or [])
            except Exception:
                lignes = []
        if not lignes:
            try:  # repli : premières pages du fichier (portée, définitions)
                r = (supabase.table(table).select("id,content,metadata")
                     .eq("fichier", chemin).limit(40).execute())  # colonne « fichier » = chemin complet, déjà indexée
                lignes = sorted(r.data or [], key=lambda l: (l["metadata"].get("page") or 0))[:NB_EXTRAITS_CIBLES]
                for l in lignes:
                    l["similarity"] = 0.80
            except Exception:
                lignes = []
        for l in lignes:
            l["cible"] = True
        st.session_state["diag_cibles"][chemin.split("/")[-1]] = len(lignes)
        sortie += lignes
    return sortie


# --- Recherche par mots-clés (complète la recherche vectorielle) ---
# Les codes et sigles (TPL6, GNF1, GB3, BBSG, C25/30...) et les titres de documents sont mal captés par les
# vecteurs. La recherche plein texte de Postgres (fonction SQL recherche_texte) les retrouve exactement.
NB_LEXICAUX = 8
MOTS_GENERIQUES = {"quelle", "quel", "quels", "quelles", "comment", "donne", "donnez", "selon", "dans", "avec",
                   "tout", "tous", "plus", "cette", "cet", "ces", "sont", "etre", "avoir", "faire", "norme",
                   "normes", "exigence", "exigences", "pour", "quoi", "pourquoi", "est", "les", "des", "une"}


def _termes_lexicaux(question: str) -> str:
    """Termes distinctifs de la question, au format « a | b | c » (uniquement a-z et 0-9, donc sans risque)."""
    mots, vus = [], set()
    for m in re.findall(r"[a-z0-9]+", _sans_accents(question)):
        if len(m) < 3 or m in MOTS_VIDES or m in MOTS_GENERIQUES or m in vus:
            continue
        vus.add(m)
        mots.append(m)
    # les termes contenant un chiffre (codes) d'abord, puis les plus longs
    mots.sort(key=lambda m: (not any(ch.isdigit() for ch in m), -len(m)))
    return " | ".join(mots[:10])


def _candidats_lexicaux(question: str) -> list:
    termes = _termes_lexicaux(question)
    if not termes:
        return []
    try:
        res = _rpc("recherche_texte", {"termes": termes, "match_count": 12}, essais=1)
        return res.data or []
    except Exception:
        return []  # fonction SQL pas encore créée ou indisponible : on continue avec la seule recherche vectorielle


# --- Recherche multi-requêtes + reclassement par l'IA (pertinence : sources compatibles avec la question) ---
NB_VARIANTES = 2          # reformulations techniques de la question
RECLASSEMENT_IA = str(st.secrets.get("RECLASSEMENT_IA", "true")).strip().lower() != "false"  # secret = "false" : désactive
TAILLE_POOL_RECLASSEMENT = 30
NB_MAX_RECLASSES = 14     # extraits transmis au modèle après reclassement


@st.cache_data(ttl=3600, max_entries=300, show_spinner=False)
def etendre_requete(question: str) -> tuple:
    """Reformulations avec le vocabulaire des normes (« acceptation » -> « critères de conformité »...).
    Lève une exception en cas d'échec (donc rien n'est mis en cache)."""
    if not RECLASSEMENT_IA or len(question.split()) > 25:
        return ()
    prompt = (
        "Tu aides à chercher dans des normes de génie civil et de matériaux (béton, granulats, chaussées, sols, "
        "ciments, aciers, adjuvants), en français.\n"
        f"Question de l'agent : \"{question}\"\n\n"
        f"Écris {NB_VARIANTES} requêtes de recherche DIFFÉRENTES qui reprennent cette question avec le vocabulaire "
        "technique employé dans les normes (ex. « acceptation » -> « critères de conformité », « critères "
        "d'identification », « essais de contrôle » ; « résistance » -> « résistance à la compression fck fcm »). "
        "Chaque requête : 6 à 14 mots, une par ligne, sans numéro, sans commentaire."
    )
    resp = client.models.generate_content(
        model=MODELE_REFORMULATION, contents=prompt, config=_config(120, MODELE_REFORMULATION, 0.2))
    variantes = []
    for ligne in (resp.text or "").splitlines():
        ligne = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ligne).strip().strip('"«» ')
        if len(ligne.split()) >= 3 and ligne.lower() != question.lower():
            variantes.append(ligne)
    return tuple(variantes[:NB_VARIANTES])


def _fusionner(listes: list) -> list:
    """Fusionne des listes d'extraits (doublons regroupés). Un extrait trouvé par plusieurs requêtes gagne en rang."""
    par_cle = {}
    for liste in listes:
        for c in liste:
            m = c["metadata"]
            cle = c.get("id") if c.get("id") is not None else (m.get("fichier"), m.get("page"), c["content"][:120])
            if cle in par_cle:
                ex = par_cle[cle]
                ex["similarity"] = max(ex["similarity"], c["similarity"])
                ex["_n"] += 1
            else:
                c["_n"] = 1
                par_cle[cle] = c
    for c in par_cle.values():
        c["similarity"] += 0.015 * (c["_n"] - 1)
    return sorted(par_cle.values(), key=lambda c: c["similarity"], reverse=True)


def reclasser_llm(question: str, pool: list):
    """Gemini Flash-Lite note chaque extrait (2 = répond directement, 1 = utile, 0 = hors sujet).
    Retourne (extraits retenus triés par pertinence, True) ou (pool, False) si le reclassement échoue."""
    if not RECLASSEMENT_IA or len(pool) < 6:
        return pool, False
    lignes = []
    for i, c in enumerate(pool, 1):
        m = c["metadata"]
        extrait = " ".join(nettoyer_remplissage(c["content"]).split())[:380]
        lignes.append(f"[{i}] ({m.get('fichier')}, p.{m.get('page')}) {extrait}")
    prompt = (
        f"Question d'un agent de laboratoire : \"{question}\"\n\n"
        "Voici des extraits de normes, numérotés. Note chacun selon son utilité pour répondre à CETTE question :\n"
        "2 = contient directement la règle, le critère, la valeur ou la définition demandée ;\n"
        "1 = utile en complément (même sujet, contexte, définition liée) ;\n"
        "0 = hors sujet (autre produit, autre essai, autre notion).\n"
        "Réponds UNIQUEMENT par un objet JSON de la forme {\"1\": 2, \"2\": 0, ...} couvrant tous les extraits.\n\n"
        "EXTRAITS :\n" + "\n".join(lignes)
    )
    try:
        resp = client.models.generate_content(
            model=MODELE_REFORMULATION, contents=prompt, config=_config(700, MODELE_REFORMULATION, 0.0))
        notes = {int(k): int(v) for k, v in re.findall(r'"?(\d+)"?\s*:\s*([012])', resp.text or "")}
    except Exception:
        return pool, False
    if len(notes) < 0.6 * len(pool):
        return pool, False  # réponse inexploitable : on garde le classement vectoriel
    for i, c in enumerate(pool, 1):
        c["note"] = 2 if c.get("cible") else notes.get(i, 1)
    tries = sorted(pool, key=lambda c: (-c["note"], -c["score"]))
    retenus = [c for c in tries if c["note"] >= 1]
    if len(retenus) < 6:
        retenus += [c for c in tries if c["note"] < 1][:6 - len(retenus)]
    return retenus[:NB_MAX_RECLASSES], True


def rechercher_chunks(question: str, seulement_principales: bool = False):
    """Recherche vectorielle en deux niveaux : les normes principales sont cherchées à part (et favorisées),
    puis complétées par le reste de la base (sauf en mode « normes principales uniquement »).
    Écarte aussi les normes supprimées de Hugging Face."""
    generaux = _candidats(question)
    try:
        variantes = etendre_requete(question)
    except Exception:
        variantes = ()
    if variantes:
        listes = [generaux]
        for v in variantes:
            try:
                listes.append(_candidats(v))
            except Exception:
                pass
        generaux = _fusionner(listes)[:60]
    principaux = _principales(question, generaux)
    if variantes:
        listes = [principaux]
        for v in variantes:
            try:
                listes.append(_principales(v, generaux))
            except Exception:
                pass
        principaux = _fusionner(listes)[:NB_CANDIDATS_PRINCIPALES + 10]

    if seulement_principales:
        candidats = list(principaux)
    else:
        candidats = list(generaux)
        deja = {(c["metadata"].get("fichier"), c["metadata"].get("page"), c["content"][:120]) for c in candidats}
        candidats += [c for c in principaux
                      if (c["metadata"].get("fichier"), c["metadata"].get("page"), c["content"][:120]) not in deja]

    candidats = _candidats_cibles(question) + candidats  # norme citée par son numéro : toujours dans la course

    # Recherche par mots-clés : fusionnée avec les résultats vectoriels
    lexicaux = _candidats_lexicaux(question)
    if lexicaux:
        par_id = {c.get("id"): c for c in candidats if c.get("id") is not None}
        base = max((c["similarity"] for c in candidats), default=0.8)
        ajoutes = 0
        for c in lexicaux:
            if c.get("id") in par_id:
                par_id[c["id"]]["similarity"] += 0.03   # trouvé par les deux méthodes : confiance accrue
                continue
            if ajoutes >= NB_LEXICAUX or (seulement_principales and not est_principale(c)):
                continue
            c["similarity"] = base - 0.02 - 0.004 * ajoutes
            c["lexical"] = True
            candidats.append(c)
            ajoutes += 1

    existants = fichiers_pdf_existants()
    if existants and layout_coherent():
        candidats = [c for c in candidats if not _chemin_norm(c) or _chemin_norm(c) in existants]
    if not candidats:
        return []

    for c in candidats:
        c["principale"] = est_principale(c)
        c["score"] = (c["similarity"] + (BONUS_PRINCIPALE if c["principale"] else 0.0)
                      + (BONUS_CIBLE if c.get("cible") else 0.0))
    candidats.sort(key=lambda c: c["score"], reverse=True)

    # Reclassement par l'IA : écarte les extraits hors sujet (sources compatibles avec la question)
    reste = candidats[TAILLE_POOL_RECLASSEMENT:]
    candidats, reclasse = reclasser_llm(question, candidats[:TAILLE_POOL_RECLASSEMENT])
    if not reclasse:
        candidats = sorted(candidats + reste, key=lambda c: c["score"], reverse=True)

    meilleur = candidats[0]["score"]
    gardes, vus = [], set()
    # Extraits du document cité par l'agent : gardés d'office, même si leur similarité est faible
    # (PDF scanné, tableaux, titre peu présent dans le texte...)
    for c in [c for c in candidats if c.get("cible")][:NB_CIBLES_GARANTIS]:
        cle = (c["metadata"].get("fichier"), c["metadata"].get("page"), c["content"][:120])
        if cle not in vus:
            vus.add(cle)
            gardes.append(c)
    for c in candidats:
        cle = (c["metadata"].get("fichier"), c["metadata"].get("page"), c["content"][:120])
        if cle in vus:
            continue  # doublon
        vus.add(cle)
        if reclasse or len(gardes) < NB_MIN_RESULTATS or c["score"] >= meilleur - ECART_PERTINENCE:
            gardes.append(c)
        if len(gardes) >= NB_MAX_RESULTATS:
            break

    # Normes principales en premier, puis regroupement par document et page
    gardes.sort(key=lambda c: (not c.get("cible"), -c.get("note", 1), not c["principale"],
                               str(c["metadata"].get("fichier")), c["metadata"].get("page") or 0))
    return gardes


def construire_prompt(question: str, chunks: list, historique: list) -> str:
    contexte_docs = "\n\n---\n\n".join(
        f"[Source {i+1} — {'DOCUMENT DEMANDÉ' if c.get('cible') else 'NORME PRINCIPALE' if c.get('principale') else 'document complémentaire'} — "
        f"{c['metadata'].get('fichier')}, page {c['metadata'].get('page')}]\n{nettoyer_remplissage(c['content'])}"
        for i, c in enumerate(chunks)
    )
    derniers = historique[-NB_ECHANGES_CONTEXTE:]
    fil = "\n".join(f"{'Agent' if h['role'] == 'user' else 'Assistant'} : {h['content'][:600]}" for h in derniers)
    bloc_historique = (f"\nContexte de la conversation (sert uniquement à comprendre les renvois comme « et pour... » ; "
                       f"n'en reprends pas les sujets si la question est autonome) :\n{fil}\n") if derniers else ""

    return f"""Tu es un assistant technique pour les agents du LPEE (laboratoire d'essais de matériaux et travaux publics).

RÈGLES DE PRÉCISION (impératives) :
1. Réponds UNIQUEMENT à partir des extraits ci-dessous. N'utilise aucune connaissance extérieure.
2. RESTE DANS LE SUJET : réponds précisément à la question posée. Utilise seulement les extraits qui la traitent
   directement ou la complètent utilement ; ignore les autres, même s'ils figurent ci-dessous (ne les résume pas,
   ne les cite pas). Sois COMPLET sur le sujet : pour chaque critère, donne la règle entière (valeurs, seuils,
   formules, conditions d'application, article ou tableau de la norme) et recopie les tableaux pertinents ; ne te
   contente pas de renvoyer à un paragraphe quand son contenu figure dans les extraits. Fusionne les extraits
   qui se complètent.
3. Reprends FIDÈLEMENT les valeurs chiffrées, unités, seuils, tolérances, formules, références de normes
   (ex. NF EN, NM, ASTM), numéros d'articles/paragraphes et conditions d'essai, exactement comme écrits.
   Ne les arrondis pas, ne les convertis pas.
4. Si des extraits se contredisent (versions/normes différentes), signale-le et cite chaque source.
5. Cite chaque affirmation avec (Source X) — fichier et page.
6. Les extraits « DOCUMENT DEMANDÉ » proviennent du document que l'agent cite par son numéro ou son titre : traite-les en priorité.
   PRIORITÉ AUX NORMES PRINCIPALES : les extraits « NORME PRINCIPALE » font référence. Appuie-toi d'abord sur eux.
   N'utilise un « document complémentaire » que pour compléter, et précise-le (« d'après un document complémentaire »).
   En cas de divergence, la norme principale prévaut ; signale l'écart. Si aucun extrait de norme principale
   ne traite la question, dis-le dans ta première phrase.
7. Si la question se limite au nom ou au numéro d'un document, présente ce document d'après les extraits
   « DOCUMENT DEMANDÉ » : émetteur, édition ou date, objet, domaine d'application, plan et contenu (chapitres,
   tableaux, catégories). N'ajoute aucun sujet des échanges précédents que la question ne mentionne pas.
8. Si la question est large ou ambiguë (ex. « critère d'acceptation »), indique en une phrase l'interprétation
   retenue (ex. critères de conformité, critères d'identification...), traite-la en détail, puis propose en
   dernière ligne 1 ou 2 précisions possibles.
9. Termine par « Points non couverts » : 3 puces au maximum, uniquement ce qui manque pour répondre à CETTE
   question (pas la liste des sujets voisins). Si rien d'important ne manque, omets cette section. Ne devine jamais.

FORMAT : commence directement par la réponse (pas d'introduction) ; la réponse à la question d'abord, puis les
détails par puces ; tableau pour les valeurs ; en dernier, « Points non couverts » s'il y a lieu.
{bloc_historique}
EXTRAITS :
{contexte_docs}

QUESTION : {question}

RÉPONSE :"""


def flux_reponse(prompt: str):
    """Générateur : affiche la réponse au fur et à mesure (1er mot en ~1 s) avec repli automatique de modèle."""
    derniere_erreur = None
    for modele in MODELES_GENERATION:
        emis = False
        try:
            for morceau in client.models.generate_content_stream(
                model=modele, contents=prompt, config=_config(4096, modele)
            ):
                if morceau.text:
                    emis = True
                    yield morceau.text
            if emis:
                return
        except Exception as e:
            derniere_erreur = e
            if emis:
                yield "\n\n⚠️ Réponse interrompue (saturation du service). Reposez la question pour la compléter."
                return
            if _erreur_temporaire(e):
                time.sleep(2)
    yield (
        "⚠️ Le service Gemini est momentanément saturé (côté Google). Les documents trouvés sont listés "
        "ci-dessous ; réessayez dans une minute.\n\n"
        f"Détail technique : `{derniere_erreur}`"
    )


# --- Téléchargement des PDF originaux (dépôt Hugging Face privé, gratuit) ---
HF_REPO_ID = st.secrets.get("HF_REPO_ID", "")
HF_TOKEN = st.secrets.get("HF_TOKEN", "")


@st.cache_data(show_spinner="Récupération du PDF...", max_entries=5)
def telecharger_pdf(chemin_relatif: str) -> bytes:
    from huggingface_hub import hf_hub_download
    chemin_local = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=chemin_relatif.replace("\\", "/"),
        repo_type="dataset",
        token=HF_TOKEN,
    )
    with open(chemin_local, "rb") as f:
        return f.read()


# --- Visionneuse PDF intégrée (page affichée en image, avec passage cité surligné) ---
@st.cache_data(show_spinner="Affichage de la page...", max_entries=60)
def rendre_page_pdf(chemin: str, page: int, zoom: float, surligner: str = "") -> tuple:
    """Retourne (image PNG de la page, nombre de pages). Surligne le passage cité si le PDF a une couche texte."""
    import fitz  # PyMuPDF
    with fitz.open(stream=telecharger_pdf(chemin), filetype="pdf") as doc:
        nb = len(doc)
        p = doc[max(1, min(nb, int(page))) - 1]
        if surligner:
            try:
                mots = surligner.split()
                for debut in (0, 6, 12):
                    morceau = " ".join(mots[debut:debut + 6])
                    if len(mots[debut:debut + 6]) >= 3:
                        for rect in p.search_for(morceau):
                            p.add_highlight_annot(rect)
            except Exception:
                pass  # PDF scanné (pas de texte) ou passage non retrouvé : on affiche simplement la page
        pix = p.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("png"), nb


def _aller_page(cle: str, delta: int):
    st.session_state[f"visu_page_{cle}"] = max(1, int(st.session_state.get(f"visu_page_{cle}", 1)) + delta)


def _entier(valeur, defaut: int = 1) -> int:
    try:
        return max(1, int(valeur))
    except (TypeError, ValueError):
        return defaut


@st.dialog("👁️ Visualisation du document", width="large")
def fenetre_visionneuse(chemin: str, nom: str, page_extrait: int, extrait: str, cle: str):
    """Fenêtre pop-up : page du PDF en image, navigation, passage cité surligné, téléchargement."""
    st.markdown(f"**📄 {nom}**")
    page = _entier(st.session_state.get(f"visu_page_{cle}", page_extrait))
    hd = bool(st.session_state.get(f"visu_hd_{cle}", False))
    try:
        surligner = extrait if page == page_extrait else ""
        png, nb = rendre_page_pdf(chemin, page, 2.6 if hd else 1.8, surligner)
    except Exception as e:
        st.error(f"Impossible d'afficher ce PDF : {e}")
        return
    st.session_state[f"visu_page_{cle}"] = max(1, min(nb, page))  # reste dans les limites du document
    c_prec, c_num, c_suiv, c_hd, c_dl = st.columns([1.3, 1.1, 1.3, 1.6, 1.6])
    c_prec.button("◀ Précédente", key=f"visu_prec_{cle}", on_click=_aller_page, args=(cle, -1),
                  disabled=page <= 1, use_container_width=True)
    c_num.number_input("Page", min_value=1, max_value=nb, step=1, key=f"visu_page_{cle}",
                       label_visibility="collapsed")
    c_suiv.button("Suivante ▶", key=f"visu_suiv_{cle}", on_click=_aller_page, args=(cle, 1),
                  disabled=page >= nb, use_container_width=True)
    c_hd.checkbox("Haute définition", key=f"visu_hd_{cle}")
    try:
        c_dl.download_button("⬇️ Télécharger", data=telecharger_pdf(chemin), file_name=nom,
                             mime="application/pdf", key=f"visu_dl_{cle}", use_container_width=True)
    except Exception:
        pass
    st.caption(f"Page {max(1, min(nb, page))} / {nb}"
               + (" · passage cité surligné" if page == page_extrait and extrait else ""))
    st.image(png, use_container_width=True)


def ouvrir_visionneuse(cle: str, chemin: str, nom: str, page, extrait: str = ""):
    page = _entier(page)
    st.session_state[f"visu_page_{cle}"] = page
    fenetre_visionneuse(chemin, nom or chemin.split("/")[-1], page, " ".join((extrait or "").split())[:300], cle)


def afficher_sources(chunks: list, prefixe_cle: str):
    existants = fichiers_pdf_existants()
    verif = bool(existants) and layout_coherent()
    peut_visualiser = bool(HF_REPO_ID and HF_TOKEN)

    def _supprime(c) -> bool:
        return verif and bool(_chemin_norm(c)) and _chemin_norm(c) not in existants

    nb_retirees = sum(1 for c in chunks if _supprime(c))
    with st.expander("📎 Sources utilisées"):
        for i, c in enumerate(chunks, 1):
            if _supprime(c):
                continue
            meta = c["metadata"]
            etiquette = "⭐ Norme principale" if c.get("principale") else "Complémentaire"
            st.markdown(f"**Source {i} · {meta.get('fichier')}** — page {meta.get('page')} · {etiquette} (pertinence : {c['similarity']:.0%})")
            st.caption(nettoyer_remplissage(c["content"]))
            if peut_visualiser and meta.get("chemin"):
                if st.button("👁️ Visualiser cette source", key=f"vis_src_{prefixe_cle}_{i}"):
                    ouvrir_visionneuse(prefixe_cle, meta["chemin"], meta.get("fichier"), meta.get("page"), c["content"])
            st.divider()

    if nb_retirees:
        st.caption(f"ℹ️ {nb_retirees} extrait(s) cité(s) proviennent de documents retirés de la base : masqués.")

    if not peut_visualiser:
        return

    fichiers_uniques = {}
    for c in chunks:
        if _supprime(c):
            continue
        meta = c["metadata"]
        if meta.get("chemin"):
            fichiers_uniques[meta["chemin"]] = meta.get("fichier")

    if fichiers_uniques:
        st.markdown("**📥 Documents sources : visualiser ou télécharger**")
        for i, (chemin, nom) in enumerate(fichiers_uniques.items()):
            cle = f"{prefixe_cle}_{i}"
            c_nom, c_vis, c_dl = st.columns([5, 2, 2], vertical_alignment="center")
            c_nom.markdown(f"📄 {nom}")
            visualiser = c_vis.button("👁️ Visualiser", key=f"vis_dl_{cle}", use_container_width=True)
            preparer = c_dl.button("⬇️ Télécharger", key=f"prep_{cle}", use_container_width=True)
            if visualiser:
                ouvrir_visionneuse(prefixe_cle, chemin, nom, 1, "")
            if preparer:
                try:
                    st.download_button(
                        f"⬇️ Enregistrer {nom}",
                        data=telecharger_pdf(chemin),
                        file_name=nom,
                        mime="application/pdf",
                        key=f"dl_{cle}",
                    )
                except Exception as e:
                    st.error(f"PDF indisponible ({nom}) : {e}")


# --- Gestion multi-conversations & historique ---
if st.session_state.get("historique_charge_pour") != st.session_state.username_courant:
    chargé = charger_historique(st.session_state.username_courant)
    if chargé:
        st.session_state.sessions, cible = chargé
        st.session_state.session_courante = cible if cible in st.session_state.sessions else list(st.session_state.sessions)[-1]
    else:
        st.session_state.sessions = {"Conversation 1": []}
        st.session_state.session_courante = "Conversation 1"
    st.session_state.historique_charge_pour = st.session_state.username_courant

historique = st.session_state.sessions[st.session_state.session_courante]

# --- Barre latérale (Sidebar) ---
VERSION_APP = "v2026.10.14-visionneuse"
with st.sidebar:
    st.caption(f"Version : {VERSION_APP}")
    if os.path.exists("logo lpee.jpg"):
        st.image("logo lpee.jpg", use_container_width=True)
    elif os.path.exists("logo_lpee.jpg"):
        st.image("logo_lpee.jpg", use_container_width=True)
    
    if EFFETS_IA:
        _nom_aff = str(st.session_state.nom_utilisateur or "")
        _initiales = "".join(m[0] for m in _nom_aff.split()[:2]).upper() or "?"
        st.markdown(
            f'''<div class="carte-ia">
  <div class="avatar-ia"><span>{_html.escape(_initiales)}</span></div>
  <div class="infos-ia">
    <div class="etat-ia"><i class="pt-ia"></i>Connecté · session sécurisée</div>
    <div class="nom-ia">{_html.escape(_nom_aff)}</div>
    <div class="eq-ia" title="Assistant IA actif"><b></b><b></b><b></b><b></b><b></b><b></b><b></b><em>IA active</em></div>
  </div>
</div>''', unsafe_allow_html=True)
    else:
        st.write(f"Connecté : **{st.session_state.nom_utilisateur}**")
    if st.button("🚪 Déconnexion", use_container_width=True):
        st.session_state.authentifie = False
        st.session_state.username_courant = ""
        st.session_state.nom_utilisateur = ""
        st.session_state.role_utilisateur = ""
        for k in ("sessions", "session_courante", "historique_charge_pour"):
            st.session_state.pop(k, None)
        st.session_state.deconnexion_volontaire = True
        try:
            try:
                controller.remove(COOKIE_NOM)
            except (TypeError, AttributeError, KeyError):
                pass
        except Exception:
            pass
        time.sleep(0.7)
        st.rerun()

    with st.expander("🔑 Changer mon mot de passe"):
        with st.form("form_mdp", clear_on_submit=True):
            ancien = st.text_input("Mot de passe actuel", type="password")
            nouveau = st.text_input("Nouveau mot de passe (8 caractères min.)", type="password")
            confirmation = st.text_input("Confirmer le nouveau mot de passe", type="password")
            valider_mdp = st.form_submit_button("Modifier", use_container_width=True)
        if valider_mdp:
            login = st.session_state.username_courant
            actuels = charger_utilisateurs()
            if actuels is None or login not in actuels:
                st.error("Impossible de vérifier votre compte, réessayez.")
            elif not verifier_mdp(ancien, actuels[login]["password"]):
                st.error("Mot de passe actuel incorrect.")
            elif len(nouveau) < 8:
                st.error("Le nouveau mot de passe doit contenir au moins 8 caractères.")
            elif nouveau != confirmation:
                st.error("La confirmation ne correspond pas.")
            elif verifier_mdp(nouveau, actuels[login]["password"]):
                st.error("Le nouveau mot de passe doit être différent de l'ancien.")
            else:
                def _changer(users):
                    users[login]["password"] = hacher_mdp(nouveau)
                ok, err = modifier_comptes(_changer)
                if ok:
                    poser_cookie(controller, login, st.session_state.utilisateurs[login]["password"])
                    st.success("✅ Mot de passe modifié. Utilisez-le à la prochaine connexion.")
                else:
                    st.error(f"Échec de l'enregistrement : {err}")

    if st.session_state.role_utilisateur == "admin":
        with st.expander("🧹 Synchronisation des normes"):
            st.caption("Les normes ajoutées sur Hugging Face sont lues (OCR si scannées) et ajoutées à l'index ; "
                       "celles supprimées en sont retirées (vérification au démarrage puis toutes les 3 h).")
            table_idx = st.secrets.get("SUPABASE_TABLE", "documents")
            _reg = charger_reglages()
            _aut = _dossiers_autorises(_reg)
            if _reg is not None and _reg.get("erreur"):
                st.error("Réglages d'indexation illisibles (Hugging Face) : aucune indexation par sécurité.")
            elif _aut is None:
                st.caption("📂 Indexation : tous les dossiers de Hugging Face")
            elif not _aut:
                st.warning("📂 Aucun dossier sélectionné : rien ne sera indexé.")
            else:
                st.caption("📂 Indexation limitée à : " + " · ".join(_aut))
            _ex_ui = fichiers_pdf_existants()
            if _secret_dossiers():
                st.info("Les dossiers à indexer sont imposés par le secret DOSSIERS_INDEXES "
                        "(supprimez-le des secrets pour choisir ici).")
            elif _ex_ui:
                _tot = {}
                for _f in _ex_ui:
                    _k = _dossier_racine(_f)
                    if _k:
                        _tot[_k] = _tot.get(_k, 0) + 1
                _idx = stats_dossiers_indexes(table_idx)
                _opts = sorted(_tot, key=str.lower)
                with st.container(border=True):
                    st.markdown("**📂 Dossiers à indexer**")
                    _tous = st.checkbox("Tous les dossiers", value=(_reg is None or bool(_reg.get("tous"))),
                                        key="reg_tous")
                    _defaut = [d for d in (_reg or {}).get("dossiers", []) if d in _opts]
                    _choix = st.multiselect("Dossiers", _opts, default=_defaut, disabled=_tous, key="reg_dossiers",
                                            format_func=lambda d: f"{d} — {_idx.get(d, 0)}/{_tot[d]} indexés",
                                            placeholder="Choisissez un ou plusieurs dossiers")
                    if st.button("💾 Enregistrer la sélection", use_container_width=True, key="reg_save"):
                        ok, msg = sauver_reglages({"tous": bool(_tous), "dossiers": [] if _tous else list(_choix)})
                        if ok:
                            st.success("Sélection enregistrée. Cliquez sur « Synchroniser maintenant » pour lancer l'indexation.")
                        else:
                            st.error("Enregistrement impossible : " + msg)
                    st.caption("Les dossiers déjà indexés qui ne sont plus cochés restent dans la base.")
            _n = nb_extraits()
            if _n is not None:
                (st.error if _n == 0 else st.caption)(f"📚 {_n:,} extraits indexés".replace(",", " ")
                                                       + (" — la base est VIDE : relancez l'indexation (ingest.py)." if _n == 0 else ""))
            _mo = taille_base_mo()
            if _mo is not None:
                _txt = f"💾 Taille de la base : {_mo:.0f} Mo / 500 Mo (offre gratuite)"
                if _mo >= 500:
                    st.error(_txt + " — base en lecture seule : supprimez des données.")
                elif _mo >= 450:
                    st.warning(_txt)
                else:
                    st.caption(_txt)
            _prog = _etat_synchro().get("progression")
            if _prog:
                _duree = int((time.time() - _etat_synchro().get("debut", time.time())) / 60)
                st.info(f"⏳ Indexation en cours (depuis {_duree} min) : " + _prog)
            with st.expander("🔎 Tester un fichier"):
                _nom_test = st.text_input("Nom (ou partie du nom) du PDF", key="diag_nom",
                                          placeholder="ex : catalogue structures chaussées")
                if _nom_test.strip():
                    fichiers_pdf_existants.clear()
                    _ex_t = fichiers_pdf_existants()
                    if not _ex_t:
                        st.error("Liste des PDF Hugging Face indisponible (vérifiez HF_REPO_ID / HF_TOKEN).")
                    else:
                        try:
                            _base_t = {_norm_chemin(c) for c in chemins_indexes(table_idx)}
                        except Exception as e:
                            _base_t = set()
                            st.error("Lecture de l'index impossible : " + str(e)[:200])
                        _mots = _sans_accents(_norm_chemin(_nom_test)).split()
                        _trouves = sorted(c for c in _ex_t if all(m in _sans_accents(c) for m in _mots))
                        _aut_t = _dossiers_autorises()
                        if not _trouves:
                            st.warning(f"Aucun PDF correspondant sur Hugging Face (dépôt « {st.secrets.get('HF_REPO_ID', '?')} », "
                                       f"{len(_ex_t)} PDF listés). Le fichier n'a pas été envoyé dans ce dépôt, "
                                       "ou son nom est différent : relancez upload_pdfs.py et vérifiez le nom sur le site.")
                        for _c in _trouves[:15]:
                            if _c in _base_t:
                                st.success(f"✅ Indexé — {_c}")
                            elif not _autorise(_c, _aut_t):
                                st.info(f"🚫 Sur Hugging Face mais son dossier n'est pas sélectionné — {_c}")
                            else:
                                st.warning(f"⏳ Sur Hugging Face, pas encore indexé (cliquez sur « Synchroniser maintenant ») — {_c}")
            if st.button("Vérifier l'état de l'index", use_container_width=True):
                fichiers_pdf_existants.clear()
                _ex = fichiers_pdf_existants()
                if not _ex:
                    st.error("Liste des PDF Hugging Face indisponible (vérifiez HF_REPO_ID / HF_TOKEN).")
                else:
                    try:
                        _en_base = {_norm_chemin(c) for c in chemins_indexes(table_idx)}
                        _aut_now = _dossiers_autorises()
                        _ex_f = {c for c in _ex if _autorise(c, _aut_now)}
                        _manq = sorted(_ex_f - _en_base)
                        st.write(f"{len(_ex)} PDF sur Hugging Face · {len(_ex_f)} dans les dossiers sélectionnés · "
                                 f"{len(_ex_f & _en_base)} indexés")
                        if _manq:
                            st.warning(f"Pas encore indexés ({len(_manq)}) : " + ", ".join(_manq[:30])
                                       + (" …" if len(_manq) > 30 else ""))
                        else:
                            st.success("Tous les PDF des dossiers sélectionnés sont indexés.")
                    except Exception as _e:
                        st.error(f"Lecture de l'index impossible : {str(_e)[:300]}")
                        if "57014" in str(_e) or "timeout" in str(_e).lower():
                            st.info("L'index Supabase est trop lent à lire : exécutez le script "
                                    "schema.sql dans Supabase > SQL Editor.")
            if st.button("Synchroniser maintenant", use_container_width=True):
                fichiers_pdf_existants.clear()
                etat = _etat_synchro()
                if not etat["verrou"].acquire(blocking=False):
                    st.info("⏳ Une indexation est déjà en cours en arrière-plan"
                            + (f" : {etat['progression']}" if etat.get("progression") else "")
                            + ". La norme sera cherchable à la fin. Cliquez sur « Vérifier l'état » pour suivre.")
                else:
                    try:
                        with st.spinner("Synchronisation (l'OCR d'une norme scannée peut prendre plusieurs minutes)..."):
                            existants = fichiers_pdf_existants()
                            r = synchroniser_index(existants, table_idx)
                            _appliquer_resultat(etat, r)
                            ri = indexer_nouveaux(existants, table_idx, charger_modele(),
                                                  st.secrets.get("HF_REPO_ID", ""), st.secrets.get("HF_TOKEN", ""),
                                                  suivi=etat, maximum=_max_manuel())
                            etat["indexation"] = ri
                    finally:
                        etat["verrou"].release()
                    if r["erreur"]:
                        st.error(r["erreur"])
                    if r["echecs"]:
                        st.warning("Suppression refusée par Supabase (droits insuffisants ?) : " + ", ".join(r["echecs"]))
                    if r["supprimes"]:
                        st.success("Retirées de l'index : " + ", ".join(r["supprimes"]))
                    if ri["erreur"]:
                        st.error(ri["erreur"])
                    if ri["indexes"]:
                        st.success("Ajoutées à l'index : " + ", ".join(ri["indexes"]))
                    for f, e in ri["echecs"].items():
                        st.warning(f"Échec d'indexation de {f} : {e}")
                    if ri.get("restants"):
                        st.info(f"📦 Il reste {ri['restants']} PDF à indexer dans les dossiers sélectionnés. "
                                f"Patientez un moment puis cliquez de nouveau sur « Synchroniser maintenant » "
                                f"(par lots de {_max_manuel()} pour ne pas surcharger le serveur).")
                    if not (r["supprimes"] or ri["indexes"] or ri["echecs"] or r["erreur"] or ri["erreur"]):
                        st.success("Index déjà à jour.")
            auto = _etat_synchro().get("resultat")
            if auto and auto["supprimes"]:
                st.caption("Dernier nettoyage auto : " + ", ".join(auto["supprimes"]))
            if auto and (auto["erreur"] or auto["echecs"]):
                st.caption("⚠️ Nettoyage auto : " + (auto["erreur"] or "suppression refusée (droits Supabase)"))
            ind = _etat_synchro().get("indexation")
            if ind and ind["indexes"]:
                st.caption("Dernière indexation auto : " + ", ".join(ind["indexes"]))
            if ind and (ind["erreur"] or ind["echecs"]):
                st.caption("⚠️ Indexation auto : " + (ind["erreur"] or "; ".join(f"{k} ({v})" for k, v in ind["echecs"].items())))

    st.markdown("---")

    # --- Section Gestion des Utilisateurs (Visible uniquement pour l'admin) ---
    if st.session_state.get("role_utilisateur") == "admin":
        with st.expander("👥 Gestion des Utilisateurs"):
            action_user = st.radio("Action", ["Ajouter", "Modifier / Supprimer"], label_visibility="collapsed")
            
            if action_user == "Ajouter":
                st.subheader("Nouvel agent")
                n_user = st.text_input("Identifiant (login)").strip().lower()
                n_pass = st.text_input("Mot de passe", type="password")
                n_nom = st.text_input("Nom complet / Rôle")
                n_role = st.selectbox("Rôle", ["agent", "admin"])

                if st.button("Enregistrer l'agent", use_container_width=True):
                    if n_user and n_pass and n_nom:
                        def _ajouter(users):
                            if n_user in users:
                                return "Cet identifiant existe déjà."
                            users[n_user] = {"password": hacher_mdp(n_pass), "nom": n_nom, "role": n_role}
                        ok, err = modifier_comptes(_ajouter)
                        if ok:
                            st.success(f"Agent {n_nom} ajouté avec succès !")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(err)
                    else:
                        st.warning("Veuillez remplir tous les champs.")

            else:
                st.subheader("Modifier ou Supprimer")
                liste_logins = list(st.session_state.utilisateurs.keys())
                sel_user = st.selectbox("Choisir un utilisateur", liste_logins)

                if sel_user:
                    fiche = st.session_state.utilisateurs[sel_user]
                    mod_nom = st.text_input("Nom complet", value=fiche["nom"])
                    mod_pass = st.text_input("Nouveau mot de passe (vide = inchangé)", type="password")
                    mod_role = st.selectbox("Rôle", ["agent", "admin"], index=0 if fiche["role"] == "agent" else 1)

                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        if st.button("Mettre à jour", use_container_width=True):
                            def _maj(users):
                                if sel_user not in users:
                                    return "Utilisateur introuvable."
                                users[sel_user]["nom"] = mod_nom
                                users[sel_user]["role"] = mod_role
                                if mod_pass:
                                    users[sel_user]["password"] = hacher_mdp(mod_pass)
                            ok, err = modifier_comptes(_maj)
                            if ok:
                                st.success("Modifications enregistrées !")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(err)
                    with col_m2:
                        if sel_user != "admin":
                            if st.button("Supprimer", use_container_width=True, type="primary"):
                                def _suppr(users):
                                    users.pop(sel_user, None)
                                ok, err = modifier_comptes(_suppr)
                                if ok:
                                    st.success("Utilisateur supprimé.")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(err)
                        else:
                            st.caption("Admin principal non supprimable.")

        st.markdown("---")

    st.header("🗂️ Historique & Sessions")
    
    if st.button("➕ Nouvelle conversation", use_container_width=True):
        nb = len(st.session_state.sessions) + 1
        nouvelle_cle = f"Conversation {nb}"
        st.session_state.sessions[nouvelle_cle] = []
        st.session_state.session_courante = nouvelle_cle
        sauvegarder_historique()
        st.rerun()

    st.markdown("---")
    st.subheader("Mes conversations")
    
    session_choisie = st.radio(
        "Sélectionner une session",
        list(st.session_state.sessions.keys()),
        index=list(st.session_state.sessions.keys()).index(st.session_state.session_courante),
        label_visibility="collapsed"
    )
    if session_choisie != st.session_state.session_courante:
        st.session_state.session_courante = session_choisie
        sauvegarder_historique()
        st.rerun()

    st.markdown("---")
    st.markdown("### 🔍 Recherches récentes (Session active)")
    questions_passees = [item["content"] for item in historique if item["role"] == "user"]
    if questions_passees:
        for q in reversed(questions_passees[-10:]):
            st.caption(f"• {q}")
    else:
        st.caption("Aucune question pour l'instant.")

if st.session_state.get("erreur_historique"):
    st.sidebar.warning("⚠️ Historique non sauvegardé. Détail de l'erreur ci-dessous :")
    st.sidebar.code(str(st.session_state["erreur_historique"])[:400], language=None)

lancer_synchro_auto()


def indicateur_ia(zone, titre: str, detail: str):
    """Affiche un indicateur animé bien visible (ou un simple message si les effets sont désactivés)."""
    if EFFETS_IA:
        zone.markdown(
            f'''<div class="ia-penser"><span class="orbe"></span><div><div class="titre">{titre}</div>
<small>{detail}</small><div class="rail"><i></i></div></div></div>''', unsafe_allow_html=True)
    else:
        zone.caption(f"⏳ {titre} — {detail}")


def flux_avec_indicateur(flux, zone):
    """Fait disparaître l'indicateur dès que le premier mot de la réponse arrive."""
    attente = True
    for morceau in flux:
        if attente:
            zone.empty()
            attente = False
        yield morceau
    if attente:
        zone.empty()


# --- Interface Principale ---
if EFFETS_IA:
    if st.session_state.pop("anim_ia", False):
        _zone_anim = st.empty()
        with _zone_anim.container():
            animation_accueil_ia(st.session_state.nom_utilisateur, nb_extraits())
        time.sleep(4.6)
        _zone_anim.empty()
    st.markdown(CSS_IA, unsafe_allow_html=True)
    injecter_fond_ia()
else:
    st.session_state.pop("anim_ia", None)
st.title("📚 Recherche des normes et fascicules techniques — LPEE")
st.markdown('<span class="badge-ia"><span class="point"></span>Assistant IA en ligne</span>', unsafe_allow_html=True)
st.subheader(f"👋 Bonjour {st.session_state.nom_utilisateur}")
st.caption(f"Session active : **{st.session_state.session_courante}** — Posez une question, puis enchaînez des questions de suivi si besoin.")

col_titre, col_bouton = st.columns([5, 1])
with col_titre:
    seulement_principales = st.checkbox(
        "🎯 Chercher uniquement dans les normes principales",
        key="seulement_principales",
        help="Décochée : les normes principales sont consultées en priorité, puis complétées par les autres documents.",
    )
with col_bouton:
    if st.button("🗑️ Vider", use_container_width=True):
        st.session_state.sessions[st.session_state.session_courante] = []
        sauvegarder_historique()
        st.rerun()

for idx, echange in enumerate(historique):
    with st.chat_message("user" if echange["role"] == "user" else "assistant"):
        st.markdown(echange["content"])
        if echange["role"] == "assistant" and echange.get("sources"):
            afficher_sources(echange["sources"], prefixe_cle=f"{st.session_state.session_courante}_msg{idx}")

question = st.chat_input("Posez votre question ou enchaînez sur la précédente...")

if question:
    historique.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        t0 = time.time()
        try:
            zone_ia = st.empty()
            indicateur_ia(zone_ia, "L'IA analyse les documents", "Recherche vectorielle dans les normes indexées")
            question_recherche = reformuler_question(question, historique[:-1])
            chunks = rechercher_chunks(question_recherche, seulement_principales)
            for _nom, _n in (st.session_state.pop("diag_cibles", None) or {}).items():
                if _n:
                    st.caption(f"🎯 Document détecté d'après votre question : {_nom} — {_n} extrait(s) retrouvé(s)")
                else:
                    st.caption(f"⚠️ Document détecté ({_nom}) mais aucun extrait lisible en base : PDF scanné ou indexation incomplète ?")
        except Exception as e:
            zone_ia.empty()
            historique.pop()  # la question n'a pas reçu de réponse : on ne la garde pas dans l'historique
            detail = str(e)
            st.error("⚠️ La recherche dans la base a échoué. Réessayez dans un instant.")
            st.code(detail[:700], language=None)
            if "57014" in detail or "timeout" in detail.lower():
                st.info("Délai dépassé côté Supabase : exécutez `schema.sql` "
                        "(index + délai d'exécution) et vérifiez que la base reste sous 500 Mo.")
            elif "aucun extrait" in detail:
                st.info("Vérifiez dans Supabase : `select count(*), count(embedding) from documents;`")
            elif "PGRST202" in detail or "Could not find" in detail:
                st.info("Fonction SQL absente : exécutez `schema.sql` dans Supabase > SQL Editor.")
            elif "read-only" in detail.lower():
                st.info("Base en lecture seule (quota de 500 Mo dépassé) : libérez de l'espace.")
            st.stop()
        t_recherche = time.time() - t0

        if not chunks:
            zone_ia.empty()
            reponse = (
                "Aucun passage trouvé dans les **normes principales** pour cette question. "
                "Décochez « Chercher uniquement dans les normes principales » pour élargir la recherche."
                if seulement_principales else
                "Aucun document pertinent trouvé pour cette question. Reformulez-la, ou vérifiez que l'indexation a bien été exécutée."
            )
            st.markdown(reponse)
            historique.append({"role": "assistant", "content": reponse, "sources": []})
            sauvegarder_historique()
        else:
            prompt = construire_prompt(question, chunks, historique[:-1])
            t1 = time.time()
            indicateur_ia(zone_ia, "L'IA rédige la réponse", f"{len(chunks)} extraits sélectionnés · analyse en cours")
            reponse = st.write_stream(flux_avec_indicateur(filtrer_remplissage_flux(flux_reponse(prompt)), zone_ia))  # affichage progressif
            t_generation = time.time() - t1
            afficher_sources(chunks, prefixe_cle=f"{st.session_state.session_courante}_msg{len(historique)}")
            st.caption(f"⏱️ Recherche : {t_recherche:.1f} s · Génération de la réponse : {t_generation:.1f} s")
            historique.append({"role": "assistant", "content": reponse, "sources": chunks})
            sauvegarder_historique()
