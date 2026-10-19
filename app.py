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


def _cle_nom(nom) -> str:
    """Clé de comparaison d'un nom de fichier : sans dossier, sans extension, sans accents ni ponctuation."""
    base = str(nom or "").replace("\\", "/").split("/")[-1]
    if base.lower().endswith(".pdf"):
        base = base[:-4]
    return " ".join(re.sub(r"[^a-z0-9]+", " ", _sans_accents(base)).split())


@st.cache_data(ttl=60, show_spinner=False)
def index_noms_pdf() -> dict:
    """{clé du nom -> chemin dans le dépôt Hugging Face}, pour retrouver les extraits indexés sans chemin."""
    existants = fichiers_pdf_existants()
    index = {}
    for chemin in sorted(existants or ()):
        index.setdefault(_cle_nom(chemin), chemin)
    return index


def _resoudre_par_nom(nom) -> str:
    """Retrouve le PDF du dépôt Hugging Face à partir d'un nom enregistré en base : nom identique, sinon
    correspondance UNIQUE quand le nom enregistré est le début ou un morceau du nom du fichier (ou l'inverse)."""
    cle = _cle_nom(nom)
    if not cle:
        return ""
    index = index_noms_pdf()
    if cle in index:
        return index[cle]
    if len(cle) < 6:
        return ""
    debuts = [ch for k, ch in index.items() if k.startswith(cle)]
    if len(debuts) == 1:
        return debuts[0]
    morceaux = [ch for k, ch in index.items() if f" {cle} " in f" {k} " or f" {k} " in f" {cle} "]
    return morceaux[0] if len(morceaux) == 1 else ""


def _chemin_norm(chunk) -> str:
    """Chemin du PDF d'un extrait. Si l'extrait a été indexé sans chemin (anciennes indexations),
    on le retrouve par le nom du fichier dans le dépôt Hugging Face ; sinon chaîne vide."""
    meta = chunk["metadata"]
    chemin = _norm_chemin(meta.get("chemin"))
    if chemin:
        return chemin
    return _resoudre_par_nom(meta.get("fichier")) if meta.get("fichier") else ""


def chemin_pour_telechargement(chunk) -> str:
    """Chemin à utiliser pour télécharger / visualiser le PDF d'un extrait ('' si introuvable)."""
    return chunk["metadata"].get("chemin") or _chemin_norm(chunk)


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
    nb_vision = 0
    with fitz.open(local) as doc:
        for num, page in enumerate(doc, 1):
            if suivi is not None:
                suivi["progression"] = f"{nom} — lecture/OCR page {num}/{len(doc)}"
            texte = _texte_page(page)
            if _vision_indexation_active() and nb_vision < MAX_PAGES_VISION_PAR_DOC and _page_riche(page):
                if suivi is not None:
                    suivi["progression"] = f"{nom} — lecture IA des tableaux/schémas page {num}/{len(doc)}"
                lu = _lire_page_objet(page)
                if lu:
                    nb_vision += 1
                    texte = (texte + "\n\n[Tableaux et schémas lus par IA]\n" + lu).strip()
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


# --- Analyse des noms de documents : type (NM, NF, ISO...), année, numéro de norme ---
TYPES_DOCUMENTS = ["NM (Maroc)", "NF / AFNOR", "ISO", "ASTM / DIN / BS", "EN (européenne)", "Guides et autres"]
_RE_TYPE_NORME = re.compile(r"\b(NM|NF|XP|FD|ISO|ASTM|DIN|BS|EN)\b[\s\-_]*(?:EN|P|ISO|IEC)?[\s\-_]*\d")
_cache_analyse = {}


def analyser_nom_document(chemin: str) -> dict:
    """Déduit du nom du fichier : type de document, année, numéro de norme (heuristique, pas de lecture du PDF)."""
    chemin = str(chemin or "")
    if chemin in _cache_analyse:
        return _cache_analyse[chemin]
    nom = chemin.split("/")[-1].rsplit(".", 1)[0]
    brut = _sans_accents(nom).upper()
    jetons = {m.group(1) for m in _RE_TYPE_NORME.finditer(brut)}
    if "NM" in jetons:
        type_doc = "NM (Maroc)"
    elif jetons & {"NF", "XP", "FD"}:
        type_doc = "NF / AFNOR"
    elif "ISO" in jetons:
        type_doc = "ISO"
    elif jetons & {"ASTM", "DIN", "BS"}:
        type_doc = "ASTM / DIN / BS"
    elif "EN" in jetons:
        type_doc = "EN (européenne)"
    else:
        type_doc = "Guides et autres"
    # année : 4 chiffres (1980-2039) qui ne suivent pas directement NM/NF/EN/ISO... (ce serait un numéro de norme)
    annees = []
    for m in re.finditer(r"(?<!\d)(19[89]\d|20[0-3]\d)(?!\d)", brut):
        if re.search(r"\b(?:NM|NF|XP|FD|ISO|ASTM|DIN|BS|EN|P)[\s\-_.]*$", brut[:m.start()]):
            continue
        annees.append((m.start(), int(m.group(1))))
    annee = max((a for _, a in annees), default=None)
    positions_annees = {pos for pos, _ in annees}
    ref = None
    if jetons:
        m = re.search(r"\bP\s?(\d{2})\s?-\s?(\d{3})(?:-(\d{1,2})(?!\d))?", brut)
        if m:
            ref = "P" + "-".join(g for g in m.groups() if g)
        else:
            for m in re.finditer(r"(?<!\d)(\d{3,5})((?:-\d{1,2}){0,2})(?!\d)", brut):
                if m.start() in positions_annees:
                    continue
                ref = m.group(1) + m.group(2)
                break
    info = {"nom": nom, "type": type_doc, "annee": annee, "ref": ref}
    _cache_analyse[chemin] = info
    return info


@st.cache_data(ttl=300, show_spinner=False)
def catalogue_documents() -> list:
    """[(chemin, type, année ou None)] pour les PDF du dépôt Hugging Face."""
    existants = fichiers_pdf_existants()
    if not existants:
        return []
    sortie = []
    for chemin in sorted(existants):
        info = analyser_nom_document(chemin)
        sortie.append((chemin, info["type"], info["annee"]))
    return sortie


# --- Recherche d'une norme par sa référence (barre latérale) ---
def chercher_par_reference(texte: str, catalogue: list, limite: int = 12):
    """Documents dont le nom contient tous les termes saisis (« 18-508 », « EN 206 », « NM 10.1.841 »...).
    Retourne (liste [(chemin, type, année)], nombre total de correspondances)."""
    jetons = re.findall(r"[a-z0-9]+", _sans_accents(texte))
    if not jetons:
        return [], 0
    cle_requete = " ".join(jetons)
    trouves = []
    for chemin, type_doc, annee in catalogue:
        cle = _cle_nom(chemin)
        jetons_nom = cle.split()
        if all(any(f == t or (len(t) >= 3 and f.startswith(t)) for f in jetons_nom) for t in jetons):
            exact = f" {cle_requete} " in f" {cle} "
            trouves.append((not exact, -(annee or 0), chemin, type_doc, annee))
    trouves.sort(key=lambda x: (x[0], x[1], x[2]))
    return [(c, t, a) for _, _, c, t, a in trouves[:limite]], len(trouves)


@st.cache_data(ttl=120, show_spinner=False)
def nb_extraits_document(chemin: str):
    """Nombre d'extraits indexés pour un PDF (None si la base ne répond pas)."""
    try:
        table = st.secrets.get("SUPABASE_TABLE", "documents")
        return supabase.table(table).select("id", count="exact").eq("fichier", chemin).limit(1).execute().count
    except Exception:
        return None


def afficher_recherche_reference():
    """Champ « référence de la norme » : trouve le PDF, indique s'il est indexé, permet de le voir ou de limiter
    les questions à ce seul document."""
    texte = st.text_input("🔎 Rechercher une norme par sa référence", key="recherche_reference",
                          placeholder="ex. NF P 18-508, EN 206, 12350-2")
    cible = st.session_state.get("document_cible")
    if cible:
        st.info(f"🎯 Questions limitées à : **{cible.split('/')[-1]}**")
        if st.button("✖ Retirer cette limitation", key="retirer_cible", use_container_width=True):
            st.session_state.pop("document_cible", None)
            st.rerun()
    if not texte.strip():
        return
    catalogue = catalogue_documents()
    if not catalogue:
        st.caption("Liste des PDF indisponible pour le moment (dépôt Hugging Face non lu).")
        return
    resultats, total = chercher_par_reference(texte, catalogue)
    if not resultats:
        st.caption(f"Aucune norme ne correspond à « {texte.strip()} ». Essayez avec le numéro seul (ex. 18-508).")
        return
    st.caption(f"{total} document(s) trouvé(s)" + (" — les 12 premiers sont affichés" if total > len(resultats) else ""))
    for i, (chemin, type_doc, annee) in enumerate(resultats):
        nom_fichier = chemin.split("/")[-1]
        n = nb_extraits_document(chemin)
        etat = f"✅ indexé ({n} extraits)" if n else ("⚠️ non indexé" if n == 0 else "")
        with st.container(border=True):
            st.markdown(f"**{nom_fichier.rsplit('.', 1)[0]}**")
            st.caption(type_doc + (f" · {annee}" if annee else "") + (f" · {etat}" if etat else ""))
            c_voir, c_cibler = st.columns(2)
            if c_voir.button("👁️ Voir", key=f"ref_vis_{i}", use_container_width=True):
                ouvrir_visionneuse(f"ref_{i}", chemin, nom_fichier, 1, "")
            if c_cibler.button("🎯 Cibler", key=f"ref_cib_{i}", use_container_width=True, disabled=(n == 0),
                               help="Limiter vos questions à ce seul document"):
                st.session_state["document_cible"] = chemin
                st.rerun()


def afficher_filtres():
    """Filtres type / année dans la barre latérale. Retourne un frozenset de chemins autorisés, ou None (pas de filtre)."""
    catalogue = catalogue_documents()
    with st.expander("🎚️ Filtres de recherche"):
        if not catalogue:
            st.info("Liste des PDF indisponible pour le moment (dépôt Hugging Face non lu).")
            if st.button("Réessayer", key="filtres_reessayer"):
                fichiers_pdf_existants.clear()
                catalogue_documents.clear()
                st.rerun()
            return None
        types_dispo = [t for t in TYPES_DOCUMENTS if any(x[1] == t for x in catalogue)]
        annees = [x[2] for x in catalogue if x[2]]
        types_sel = st.multiselect("Type de document", types_dispo, key="filtre_types", placeholder="Tous les types")
        plage = None
        if annees and min(annees) < max(annees):
            plage = st.slider("Année du document (d'après son nom)", min(annees), max(annees),
                              (min(annees), max(annees)), key="filtre_annees")
        sans_annee = st.checkbox("Inclure les documents sans année dans leur nom", value=True, key="filtre_sans_annee")
        annee_filtree = bool(plage) and plage != (min(annees), max(annees))
        if not (types_sel or annee_filtree or not sans_annee):
            st.caption(f"Aucun filtre : les {len(catalogue)} documents sont consultés.")
            return None
        autorises = frozenset(
            chemin for chemin, type_doc, annee in catalogue
            if (not types_sel or type_doc in types_sel)
            and ((annee is None and sans_annee) or (annee is not None and (not plage or plage[0] <= annee <= plage[1]))))
        st.caption(f"🎚️ {len(autorises)} document(s) sur {len(catalogue)} correspondent aux filtres.")
        return autorises


def alertes_editions(chunks: list) -> list:
    """Messages d'alerte quand une norme citée existe en plusieurs éditions dans la base
    (ex. NF EN 13808 de 2013 et NM EN 13808 de 2017). Comparaison des noms de fichiers uniquement."""
    existants = fichiers_pdf_existants()
    if not existants:
        return []
    cites = {}
    for c in chunks:
        chemin = _chemin_norm(c)
        if chemin:
            cites[chemin] = analyser_nom_document(chemin)
    par_ref = {}
    for chemin in existants:
        info = analyser_nom_document(chemin)
        if info["ref"]:
            par_ref.setdefault(info["ref"], []).append((chemin, info))

    def libelle(info):
        if not info["annee"]:
            return f"{info['nom']} (année non précisée)"
        return info["nom"] if str(info["annee"]) in info["nom"] else f"{info['nom']} ({info['annee']})"

    messages, deja = [], set()
    for chemin, info in cites.items():
        ref = info["ref"]
        if not ref or ref in deja or len(par_ref.get(ref, [])) < 2:
            continue
        deja.add(ref)
        groupe = sorted(par_ref[ref], key=lambda x: (x[1]["annee"] or 0, x[0]))
        cites_ref = [i for ch, i in cites.items() if i["ref"] == ref]
        annees = [i["annee"] for _, i in groupe if i["annee"]]
        plus_recente = max(annees) if annees else None
        ancienne_citee = plus_recente and any(i["annee"] and i["annee"] < plus_recente for i in cites_ref)
        texte = (f"⚠️ **Plusieurs éditions pour la norme n° {ref}** dans la base : "
                 + " ; ".join(libelle(i) for _, i in groupe) + ". "
                 f"Extraits cités issus de : " + ", ".join(libelle(i) for i in cites_ref) + ". ")
        if ancienne_citee:
            texte += (f"**Une édition plus récente ({plus_recente}) existe** : cette norme peut avoir été remplacée, "
                      "vérifiez l'édition en vigueur avant d'appliquer une exigence.")
        else:
            texte += "Vérifiez l'édition en vigueur avant d'appliquer une exigence."
        messages.append(texte)
    return messages


# --- Recherche multi-requêtes + reclassement par l'IA (pertinence : sources compatibles avec la question) ---
NB_VARIANTES = 2          # reformulations techniques de la question
RECLASSEMENT_IA = str(st.secrets.get("RECLASSEMENT_IA", "true")).strip().lower() != "false"  # secret = "false" : désactive
TAILLE_POOL_RECLASSEMENT = 30
NB_MAX_RECLASSES = 14     # extraits transmis au modèle après reclassement


@st.cache_data(ttl=3600, max_entries=300, show_spinner=False)
def etendre_requete(question: str) -> tuple:
    """Reformulations de la question avec le vocabulaire des normes (synonymes, termes normatifs).
    Lève une exception en cas d'échec (donc rien n'est mis en cache)."""
    if not RECLASSEMENT_IA or len(question.split()) > 25:
        return ()
    prompt = (
        "Tu aides à chercher dans une base de normes et de documents techniques de génie civil, de géotechnique, "
        "de routes et de matériaux de construction, en français.\n"
        f"Question de l'agent : \"{question}\"\n\n"
        f"Écris {NB_VARIANTES} requêtes de recherche DIFFÉRENTES qui reprennent le même besoin avec le vocabulaire "
        "employé dans les normes : termes normatifs et synonymes (exigences, spécifications, conformité, méthode "
        "d'essai, mode opératoire, seuil, tolérance, domaine d'application, définition...), sigles, grandeurs et "
        "unités concernés. Reste fidèle au sujet de la question : n'ajoute aucun autre sujet. "
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


@st.cache_data(ttl=3600, max_entries=200, show_spinner=False)
def _candidats_filtres(question: str, fichiers: tuple):
    """Recherche vectorielle limitée à une liste de documents (filtres type / année).
    Filtre étroit : fonction SQL match_documents_fichiers (calcul exact sur le sous-ensemble).
    Filtre large (>= 40 % des documents) : recherche globale rapide (index) puis filtrage."""
    autorises = set(fichiers)
    total = len(fichiers_pdf_existants() or ())

    def via_global():
        return [c for c in _candidats(question) if _chemin_norm(c) in autorises]

    def via_sql():
        res = _rpc("match_documents_fichiers", {
            "query_embedding": _vecteur_question(question),
            "match_count": NB_CANDIDATS,
            "fichiers": list(fichiers),
        })
        return res.data or []

    def via_chemins():  # 1 à 5 documents : fonction SQL déjà en place (filtre sur le chemin)
        lignes = []
        for chemin in fichiers:
            res = _rpc("match_documents_prefixe", {
                "query_embedding": _vecteur_question(question),
                "match_count": NB_CANDIDATS,
                "motif": chemin,
            })
            lignes += res.data or []
        return lignes

    if len(autorises) <= 5:
        ordre = (via_chemins, via_sql, via_global)
    elif total and len(autorises) >= 0.4 * total:
        ordre = (via_global, via_sql)
    else:
        ordre = (via_sql, via_global)
    donnees = []
    for methode in ordre:
        try:
            donnees = methode()
        except Exception:
            donnees = []
        if donnees:
            break
    if not donnees:
        raise LookupError("Aucun extrait dans les documents sélectionnés par les filtres.")
    return sorted(donnees, key=lambda c: c["similarity"], reverse=True)


# --- Recherche par titre de document : le sens du NOM du fichier rapproche la question du bon document ---
# (ex. « ajouts de calcaire » -> « NF P 18-508 additif calcaires », même si le texte du PDF emploie d'autres mots)
NB_DOCS_TITRE = 2
SEUIL_TITRE = 0.78
MARGE_TITRE = 0.03
BONUS_TITRE = 0.02


def _titre_lisible(chemin: str) -> str:
    dossier, _, nom = chemin.rpartition("/")
    nom = " ".join(nom.rsplit(".", 1)[0].replace("_", " ").split())
    return f"{nom} ({dossier.split('/')[0]})" if dossier else nom


@st.cache_resource(ttl=7200, show_spinner="Préparation de la recherche par titre de document (une seule fois)...")
def _vecteurs_titres(chemins: tuple):
    import numpy as np
    textes = [f"passage: {_titre_lisible(c)}" for c in chemins]
    return np.asarray(charger_modele().encode(textes, normalize_embeddings=True, batch_size=64,
                                             show_progress_bar=False), dtype="float32")


def documents_proches_titre(question: str, autorises=None) -> list:
    existants = fichiers_pdf_existants()
    if not existants:
        return []
    import numpy as np
    chemins = tuple(sorted(existants))
    sims = _vecteurs_titres(chemins) @ np.asarray(_vecteur_question(question), dtype="float32")
    retenus, meilleur = [], None
    for i in np.argsort(-sims)[:40]:
        if autorises and chemins[i] not in autorises:
            continue
        s = float(sims[i])
        if meilleur is None:
            if s < SEUIL_TITRE:
                return []
            meilleur = s
        if s < meilleur - MARGE_TITRE or len(retenus) >= NB_DOCS_TITRE:
            break
        retenus.append(chemins[i])
    return retenus


def _candidats_documents(question: str, autorises=None) -> list:
    sortie = []
    for chemin in documents_proches_titre(question, autorises):
        try:
            res = _rpc("match_documents_prefixe", {
                "query_embedding": _vecteur_question(question),
                "match_count": 12,
                "motif": chemin,
            })
            lignes = _diversifier(res.data or [], n=5)
        except Exception:
            lignes = []
        for l in lignes:
            l["titre_proche"] = True
        sortie += lignes
    return sortie


def rechercher_chunks(question: str, seulement_principales: bool = False, fichiers_autorises=None):
    """Recherche vectorielle en deux niveaux : les normes principales sont cherchées à part (et favorisées),
    puis complétées par le reste de la base (sauf en mode « normes principales uniquement »).
    Écarte aussi les normes supprimées de Hugging Face."""
    autorises = fichiers_autorises if fichiers_autorises else None  # frozenset de chemins, ou None = pas de filtre
    if autorises:
        liste_autorisee = tuple(sorted(autorises))

        def chercher(q):
            return _candidats_filtres(q, liste_autorisee)
    else:
        chercher = _candidats
    try:
        generaux = chercher(question)
    except LookupError:
        if not autorises:
            raise
        generaux = []
    try:
        variantes = () if _fichiers_cibles(question) else etendre_requete(question)  # document nommé : inutile
    except Exception:
        variantes = ()
    if variantes:
        listes = [generaux]
        for v in variantes:
            try:
                listes.append(chercher(v))
            except Exception:
                pass
        generaux = _fusionner(listes)[:60]
    principaux = _principales(question, generaux)
    if autorises:
        principaux = [c for c in principaux if _chemin_norm(c) in autorises]
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

    # Documents dont le TITRE correspond à la question (sauf si la question nomme déjà un document)
    try:
        if not _fichiers_cibles(question):
            ids_connus = {c.get("id") for c in candidats if c.get("id") is not None}
            for c in _candidats_documents(question, autorises):
                if c.get("id") in ids_connus or (seulement_principales and not est_principale(c)):
                    continue
                c["similarity"] += BONUS_TITRE
                candidats.append(c)
    except Exception:
        pass  # la recherche par titre est un plus : elle ne doit jamais bloquer la recherche

    existants = fichiers_pdf_existants()
    if existants and layout_coherent():
        candidats = [c for c in candidats if not _chemin_norm(c) or _chemin_norm(c) in existants]
    if autorises:
        candidats = [c for c in candidats if _chemin_norm(c) in autorises]  # filtres type / année
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


# --- Rôle donné au modèle Gemini : ingénieur calculateur rigoureux (repris dans les prompts de réponse et de conformité) ---
ROLE_INGENIEUR = (
    "RÔLE : tu es un ingénieur calculateur senior en génie civil, géotechnique et matériaux, rigoureux et "
    "méthodique, spécialiste des essais de laboratoire et des normes (NM, NF, EN, ISO, ASTM). Tu raisonnes comme "
    "un vérificateur de notes de calcul : tu contrôles chaque valeur, chaque unité et chaque formule ; tu refais "
    "les calculs étape par étape (formule, application numérique, résultat avec unité) ; tu respectes les arrondis "
    "et chiffres significatifs imposés par la norme ; tu signales toute incohérence (unités, ordre de grandeur, "
    "valeur aberrante) ; et tu écris « non déterminable avec les extraits fournis » plutôt que d'estimer."
)


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

    return f"""{ROLE_INGENIEUR}

MISSION : tu assistes les agents du LPEE (laboratoire d'essais de matériaux et travaux publics).

RÈGLES DE PRÉCISION (impératives) :
1. Réponds UNIQUEMENT à partir des extraits ci-dessous. N'utilise aucune connaissance extérieure.
2. RESTE DANS LE SUJET : réponds précisément à la question posée, quel que soit le domaine. Utilise seulement
   les extraits qui la traitent directement ou la complètent utilement ; ignore les autres, même s'ils figurent
   ci-dessous (ne les résume pas, ne les cite pas). Sois COMPLET sur le sujet : donne la règle ou la méthode
   entière (valeurs, seuils, formules, étapes, conditions d'application, article ou tableau de la norme) et
   recopie les tableaux pertinents ; ne te contente pas de renvoyer à un paragraphe quand son contenu figure dans
   les extraits. Fusionne les extraits qui se complètent. Si la question demande explicitement une vue d'ensemble
   (présenter, résumer, lister toutes les exigences), élargis en conséquence.
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
8. Si la question est large ou ambiguë (plusieurs sens possibles, plusieurs normes ou produits concernés),
   indique en une phrase l'interprétation retenue, traite-la en détail, puis propose en dernière ligne 1 ou 2
   précisions possibles.
9. Termine par « Points non couverts » : 3 puces au maximum, uniquement ce qui manque pour répondre à CETTE
   question (pas la liste des sujets voisins). Si rien d'important ne manque, omets cette section. Ne devine jamais.
10. Les passages précédés de « [Tableaux et schémas lus par IA] » sont une transcription automatique : utilise-les, mais pour une valeur critique invite l'agent à la vérifier sur la page (bouton « Visualiser »).
11. CALCULS : si la question implique un calcul, donne la formule telle qu'elle figure dans les extraits, les données avec leurs unités, l'application numérique, puis le résultat arrondi comme l'impose la norme ; vérifie l'homogénéité des unités. Si une donnée manque, demande-la au lieu de la supposer.

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


# ==========================================
# GEMINI VISION (TABLEAUX ET SCHÉMAS) + ASSISTANT DE CONFORMITÉ
# ==========================================
# Secrets optionnels :
#   VISION_INDEXATION = "true"      -> à l'indexation, les pages à tableaux/schémas sont lues par Gemini (coûteux : désactivé par défaut)
#   MAX_PAGES_VISION_PAR_DOC = "25" -> plafond de pages lues par document à l'indexation
VISION_INDEXATION = str(st.secrets.get("VISION_INDEXATION", "false")).strip().lower() in ("true", "1", "oui", "yes")
try:
    MAX_PAGES_VISION_PAR_DOC = max(1, int(st.secrets.get("MAX_PAGES_VISION_PAR_DOC", 25)))
except Exception:
    MAX_PAGES_VISION_PAR_DOC = 25


def _generer(contenu, max_tokens: int = 4096, json_sortie: bool = False, reflexion: int = 0, temperature: float = 0.0) -> str:
    """Appel Gemini avec repli automatique de modèle. 'contenu' = texte ou liste [image/PDF, texte]."""
    derniere = None
    for modele in MODELES_GENERATION:
        cfg = {"temperature": temperature, "max_output_tokens": max_tokens}
        if json_sortie:
            cfg["response_mime_type"] = "application/json"
        if "2.5" in modele:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=reflexion)
        try:
            resp = client.models.generate_content(model=modele, contents=contenu,
                                                  config=types.GenerateContentConfig(**cfg))
            if resp.text:
                return resp.text
        except Exception as e:
            derniere = e
            if _erreur_temporaire(e):
                time.sleep(2)
    raise RuntimeError(f"Gemini indisponible : {str(derniere)[:250]}")


def _json_depuis_texte(texte: str):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (texte or "").strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"[\[{].*[\]}]", t, re.S)
        if not m:
            raise ValueError("Réponse de l'IA non exploitable (JSON attendu).")
        return json.loads(m.group(0))


# --- 1) Lecture des tableaux et schémas par Gemini (vision) ---
PROMPT_VISION = (
    "Tu analyses l'IMAGE d'une page d'une norme ou d'un document technique (génie civil, géotechnique, routes, "
    "matériaux). Ta mission : rendre lisibles par un moteur de recherche les éléments que le texte brut perd.\n"
    "1) TABLEAUX : transcris chacun fidèlement en tableau Markdown (titre du tableau, en-têtes, unités, notes de "
    "bas de tableau, valeurs EXACTEMENT comme écrites ; répète les cellules fusionnées).\n"
    "2) SCHÉMAS, FIGURES, COURBES, CROQUIS, ABAQUES : donne le titre ou la légende, ce que la figure représente, "
    "les éléments annotés, les dimensions et valeurs lisibles, les axes avec leurs unités et la tendance d'une courbe.\n"
    "3) Ne recopie pas le texte courant. N'invente rien : écris [illisible] quand c'est le cas.\n"
    "4) Si la page ne contient ni tableau ni schéma, réponds exactement : AUCUN.\n"
    "Réponds en français, sans introduction."
)


@st.cache_data(ttl=86400, max_entries=200, show_spinner=False)
def lire_page_gemini(chemin: str, page: int) -> str:
    """Transcrit les tableaux et décrit les schémas d'une page PDF (à la demande). Les erreurs ne sont pas mises en cache."""
    import fitz  # PyMuPDF
    with fitz.open(stream=telecharger_pdf(chemin), filetype="pdf") as doc:
        p = doc[max(1, min(len(doc), int(page))) - 1]
        png = p.get_pixmap(matrix=fitz.Matrix(2.2, 2.2), alpha=False).tobytes("png")
    texte = _generer([types.Part.from_bytes(data=png, mime_type="image/png"), PROMPT_VISION], 4096)
    return nettoyer_remplissage(texte.strip())


def bloc_lecture_vision(cle: str, chemin: str, page):
    """Bouton « Lire les tableaux et schémas » sous une source ; le résultat reste affiché après les rafraîchissements."""
    ck = f"vision_{cle}"
    if st.button("🧠 Lire les tableaux et schémas de cette page (IA)", key=f"btn_{ck}",
                 help="Gemini regarde la page comme une image : tableaux transcrits, schémas décrits."):
        with st.spinner("Gemini analyse la page..."):
            try:
                st.session_state[ck] = lire_page_gemini(chemin, _entier(page))
            except Exception as e:
                st.session_state[ck] = f"⚠️ Lecture impossible : {str(e)[:200]}"
    if st.session_state.get(ck):
        st.markdown(st.session_state[ck])
        st.caption("Transcription automatique par IA : vérifiez les valeurs critiques sur la page "
                   "(bouton « Visualiser cette source »).")


def _vision_indexation_active() -> bool:
    return VISION_INDEXATION


def _page_riche(page) -> bool:
    """Page contenant un tableau, une image ou beaucoup de tracés vectoriels (schéma)."""
    try:
        if page.find_tables().tables:
            return True
    except Exception:
        pass
    try:
        if len(page.get_images()) >= 1 or len(page.get_drawings()) >= 30:
            return True
    except Exception:
        pass
    return False


def _lire_page_objet(page) -> str:
    """Lecture Gemini d'une page pendant l'indexation. Chaîne vide si rien d'utile ou en cas d'échec."""
    try:
        png = page.get_pixmap(dpi=150).tobytes("png")
        texte = _generer([types.Part.from_bytes(data=png, mime_type="image/png"), PROMPT_VISION], 3000)
    except Exception:
        return ""
    texte = nettoyer_remplissage((texte or "").strip())
    return "" if (not texte or texte.upper().startswith("AUCUN")) else texte


# --- 2) Assistant de conformité ---
MODE_RECHERCHE = "💬 Recherche dans les normes"
MODE_CONFORMITE = "✅ Assistant de conformité"
MAX_PARAMETRES_CONFORMITE = 6
MAX_EXTRAITS_CONFORMITE = 24
MAX_OCTETS_FICHIER = 15 * 1024 * 1024
VERDICTS = {"conforme": "✅ Conforme", "non_conforme": "❌ Non conforme", "indetermine": "❓ Indéterminé"}

PROMPT_EXTRACTION_FICHIER = (
    "Tu lis un rapport ou procès-verbal d'essai de laboratoire (génie civil, matériaux, géotechnique, routes). "
    "Transcris TOUS les résultats d'essai, un par ligne, au format :\n"
    "paramètre | valeur | unité | méthode ou norme d'essai (si indiquée) | échantillon, âge ou conditions (si indiqués)\n"
    "Inclus aussi, en tête, l'objet de l'essai et la classe ou catégorie visée si le document les mentionne. "
    "Reprends les valeurs EXACTEMENT comme écrites. N'invente rien ; écris [illisible] si besoin. "
    "Ne commente pas, n'interprète pas, ne conclus pas sur la conformité."
)

PROMPT_MESURES = (
    "Voici des résultats d'essai saisis par un agent de laboratoire :\n\n__RESULTATS__\n\n"
    "Extrais chaque grandeur mesurée ou caractéristique à vérifier (12 au maximum). Réponds UNIQUEMENT par un "
    "tableau JSON : [{\"parametre\": \"...\", \"valeur\": \"...\", \"unite\": \"...\", \"methode\": \"...\", "
    "\"conditions\": \"...\"}]. Garde les valeurs telles qu'écrites (plusieurs éprouvettes ou un intervalle = "
    "une seule entrée). Champ inconnu = chaîne vide. N'invente rien."
)

PROMPT_CONFORMITE = ROLE_INGENIEUR + """

MISSION : tu es l'assistant de conformité du LPEE (laboratoire d'essais de matériaux et travaux publics). Tu compares des RÉSULTATS D'ESSAI aux EXIGENCES figurant dans les extraits de normes ci-dessous.

RÈGLES IMPÉRATIVES :
1. Les exigences viennent UNIQUEMENT des extraits. Aucune connaissance extérieure. Si l'exigence n'y figure pas : verdict "indetermine".
2. "conforme" seulement si la valeur satisfait EXPLICITEMENT l'exigence (seuil, plage, classe) ; "non_conforme" si elle ne la satisfait pas ; "indetermine" si l'exigence est absente, dépend d'une classe ou d'une condition non précisée, si les unités sont incompatibles ou si la valeur est illisible.
3. Compare les nombres avec rigueur (≥, ≤, <, >, plages, tolérances, unités). Conversion d'unité seulement si triviale (ex. kPa/MPa), et dis-le.
4. Cite l'exigence telle qu'écrite (valeur, unité, article ou tableau) et la source (numéro de source).
5. Si l'exigence varie selon une classe ou une catégorie non précisée par l'agent, donne les cas dans "exigence" et mets "indetermine".
6. Ne tiens pas compte de l'incertitude de mesure sauf si l'extrait l'impose ; signale dans le commentaire qu'une valeur à moins de 5 % du seuil mérite vérification.
7. Dans "points_a_verifier", signale : éditions différentes ou contradictoires, passages marqués « lus par IA » (transcription automatique à vérifier sur la page), méthode d'essai de l'agent différente de celle de la norme, informations manquantes pour conclure.
8. Ne devine jamais. Sois concis.
9. CALCULS : avant de comparer, recalcule toute grandeur dérivée (moyenne, écart, rapport, résistance à partir d'une force et d'une section...) avec la formule de la norme si elle figure dans les extraits ; indique le calcul dans "commentaire" ; signale toute incohérence entre les valeurs saisies (unités, ordre de grandeur).

Réponds UNIQUEMENT par un objet JSON :
{"synthese": "2 phrases maximum", "criteres": [{"parametre": "...", "valeur_mesuree": "valeur et unité", "exigence": "...", "verdict": "conforme|non_conforme|indetermine", "ecart": "marge ou écart chiffré, sinon vide", "sources": [numéros de source], "commentaire": "..."}], "points_a_verifier": ["..."]}

__ENTREE__

EXTRAITS :
__EXTRAITS__
"""


def lire_fichier_resultats(nom: str, octets: bytes, mime: str) -> str:
    """Extrait les résultats d'essai d'un PV (PDF, image ou texte) avec Gemini."""
    if len(octets) > MAX_OCTETS_FICHIER:
        raise ValueError("Fichier trop volumineux (15 Mo maximum).")
    if (mime or "").startswith("text/") or nom.lower().endswith((".txt", ".csv")):
        return octets.decode("utf-8", errors="replace").strip()
    if nom.lower().endswith(".pdf"):
        mime = "application/pdf"
    elif nom.lower().endswith((".jpg", ".jpeg")):
        mime = "image/jpeg"
    elif nom.lower().endswith(".png"):
        mime = "image/png"
    return _generer([types.Part.from_bytes(data=octets, mime_type=mime), PROMPT_EXTRACTION_FICHIER], 4096).strip()


def extraire_mesures(resultats: str) -> list:
    brut = _generer(PROMPT_MESURES.replace("__RESULTATS__", resultats[:8000]), 2048, json_sortie=True)
    data = _json_depuis_texte(brut)
    if isinstance(data, dict):
        data = data.get("mesures") or data.get("resultats") or list(data.values())[0]
    return [m for m in data if isinstance(m, dict) and m.get("parametre")][:12]


def _cle_extrait(c) -> tuple:
    return (c["metadata"].get("fichier"), c["metadata"].get("page"), c["content"][:120])


def rassembler_exigences(mesures: list, referentiel: str, contexte: str, seulement_principales, fichiers_autorises) -> list:
    """Cherche dans la base les exigences pour chaque paramètre mesuré, puis fusionne les extraits."""
    requetes = []
    for m in mesures[:MAX_PARAMETRES_CONFORMITE]:
        requetes.append(" ".join(x for x in (referentiel, m.get("parametre"), m.get("methode"),
                                             "exigence spécification critère de conformité seuil", contexte) if x))
    reste = [m.get("parametre", "") for m in mesures[MAX_PARAMETRES_CONFORMITE:]]
    if reste:
        requetes.append(" ".join(x for x in (referentiel, " ".join(reste), "exigences spécifications", contexte) if x))
    groupes, derniere_erreur = [], None
    for q in requetes:
        try:
            groupes.append(rechercher_chunks(q, seulement_principales, fichiers_autorises)[:8])
        except Exception as e:
            derniere_erreur = e
    st.session_state.pop("diag_cibles", None)
    if not any(groupes):
        if derniere_erreur:
            raise derniere_erreur
        return []
    fusion, vus = [], set()
    for rang in range(8):  # tour de rôle entre paramètres : chacun garde ses meilleurs extraits
        for g in groupes:
            if rang < len(g) and _cle_extrait(g[rang]) not in vus:
                vus.add(_cle_extrait(g[rang]))
                fusion.append(g[rang])
    return fusion[:MAX_EXTRAITS_CONFORMITE]


def verifier_conformite(mesures: list, chunks: list, referentiel: str, contexte: str) -> dict:
    extraits = "\n\n---\n\n".join(
        f"[Source {i+1} — {'NORME PRINCIPALE' if c.get('principale') else 'document complémentaire'} — "
        f"{c['metadata'].get('fichier')}, page {c['metadata'].get('page')}]\n{nettoyer_remplissage(c['content'])}"
        for i, c in enumerate(chunks))
    lignes = "\n".join(
        "- " + " | ".join(str(m.get(k, "")).strip() for k in ("parametre", "valeur", "unite", "methode", "conditions"))
        for m in mesures)
    entree = ("RÉSULTATS À VÉRIFIER (paramètre | valeur | unité | méthode | conditions) :\n" + lignes
              + (f"\nRéférentiel indiqué par l'agent : {referentiel}" if referentiel else "")
              + (f"\nProduit, classe ou usage visé : {contexte}" if contexte else ""))
    prompt = PROMPT_CONFORMITE.replace("__ENTREE__", entree).replace("__EXTRAITS__", extraits)
    data = _json_depuis_texte(_generer(prompt, 6000, json_sortie=True, reflexion=1024))
    if not isinstance(data, dict) or not isinstance(data.get("criteres"), list):
        raise ValueError("Réponse de l'IA incomplète : relancez la vérification.")
    return data


def _noms_sources(nums, chunks: list) -> str:
    if not isinstance(nums, list):
        nums = [nums]
    sortie = []
    for n in nums:
        try:
            m = chunks[int(n) - 1]["metadata"]
            sortie.append(f"{m.get('fichier')} p.{m.get('page')}")
        except Exception:
            continue
    return " ; ".join(sortie)


def afficher_conformite(res: dict):
    data, chunks = res["verdict"], res["chunks"]
    criteres = data["criteres"]
    nb = {k: sum(1 for c in criteres if c.get("verdict") == k) for k in VERDICTS}
    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Conformes", nb["conforme"])
    c2.metric("❌ Non conformes", nb["non_conforme"])
    c3.metric("❓ Indéterminés", nb["indetermine"])
    if data.get("synthese"):
        (st.error if nb["non_conforme"] else st.success if nb["conforme"] and not nb["indetermine"] else st.warning)(
            data["synthese"])
    tableau = [{
        "Paramètre": c.get("parametre", ""),
        "Valeur mesurée": c.get("valeur_mesuree", ""),
        "Exigence de la norme": c.get("exigence", ""),
        "Verdict": VERDICTS.get(c.get("verdict"), "❓ Indéterminé"),
        "Écart / marge": c.get("ecart", ""),
        "Sources": _noms_sources(c.get("sources", []), chunks),
        "Commentaire": c.get("commentaire", ""),
    } for c in criteres]
    st.dataframe(tableau, use_container_width=True, hide_index=True)
    if data.get("points_a_verifier"):
        st.markdown("**⚠️ Points à vérifier**")
        for p in data["points_a_verifier"]:
            st.markdown(f"- {p}")
    st.caption("Aide à la décision : la conclusion doit être validée par le responsable technique, "
               "sur l'édition de la norme en vigueur.")
    if tableau:
        import csv
        tampon = io.StringIO()
        w = csv.DictWriter(tampon, fieldnames=list(tableau[0].keys()))
        w.writeheader()
        w.writerows(tableau)
        st.download_button("⬇️ Exporter le tableau (CSV)", tampon.getvalue().encode("utf-8-sig"),
                           "verification_conformite.csv", "text/csv", key="conf_export_csv")
    afficher_sources(chunks, prefixe_cle="conf")


def page_conformite(fichiers_autorises=None):
    """Page « Assistant de conformité » : résultats d'essai -> exigences des normes -> verdict par critère."""
    if EFFETS_IA:
        st.markdown(CSS_IA, unsafe_allow_html=True)
        injecter_fond_ia()
    st.title("✅ Assistant de conformité — LPEE")
    st.caption("Saisissez (ou chargez) des résultats d'essai : l'assistant retrouve les exigences dans les normes "
               "indexées et indique, critère par critère, si les résultats sont conformes, non conformes ou "
               "indéterminés, avec la source de chaque exigence.")
    if fichiers_autorises is not None and not fichiers_autorises:
        st.warning("Aucun document ne correspond aux filtres choisis : élargissez-les dans la barre latérale.")
        return
    if st.session_state.get("document_cible"):
        st.info(f"🎯 Exigences cherchées uniquement dans : **{st.session_state['document_cible'].split('/')[-1]}**")
    elif fichiers_autorises is not None:
        st.info(f"🎚️ Filtres actifs : exigences cherchées dans {len(fichiers_autorises)} document(s).")

    with st.expander("📄 Charger un PV ou rapport d'essai (PDF, photo, texte)"):
        fichier = st.file_uploader("Fichier", type=["pdf", "png", "jpg", "jpeg", "txt", "csv"], key="conf_fichier")
        if fichier is not None and st.button("🧠 Extraire les résultats avec Gemini", key="conf_extraire"):
            with st.spinner("Gemini lit le document..."):
                try:
                    st.session_state["conf_resultats"] = lire_fichier_resultats(fichier.name, fichier.getvalue(), fichier.type)
                    st.success("Résultats extraits : relisez-les et corrigez-les au besoin ci-dessous.")
                except Exception as e:
                    st.error(f"Lecture impossible : {str(e)[:250]}")

    resultats = st.text_area("Résultats d'essai à vérifier", key="conf_resultats", height=170,
                             placeholder="ex. Béton C25/30 — résistance en compression à 28 j : 31,2 MPa (NM EN 12390-3) ; "
                                         "affaissement : 90 mm (NM EN 12350-2) ; teneur en air : 4,5 %")
    c_ref, c_ctx = st.columns(2)
    referentiel = c_ref.text_input("Norme ou référentiel de comparaison (facultatif)", key="conf_ref",
                                   placeholder="ex. NF EN 206/CN, NM 10.1.008")
    contexte = c_ctx.text_input("Produit, classe ou usage visé (facultatif)", key="conf_ctx",
                                placeholder="ex. béton C25/30 XC2, GNF1 en couche de forme")

    if st.button("✅ Vérifier la conformité", type="primary", disabled=not resultats.strip(), key="conf_lancer"):
        st.session_state.pop("conf_resultat", None)
        with st.status("Analyse en cours...", expanded=True) as etat:
            try:
                st.write("1/3 · Lecture des résultats saisis")
                mesures = extraire_mesures(resultats)
                if not mesures:
                    raise ValueError("Aucune mesure exploitable détectée : précisez les paramètres et leurs valeurs.")
                st.write(f"2/3 · Recherche des exigences dans les normes ({len(mesures)} paramètre(s))")
                chunks = rassembler_exigences(mesures, referentiel.strip(), contexte.strip(),
                                              st.session_state.get("seulement_principales", False), fichiers_autorises)
                if not chunks:
                    raise ValueError("Aucune exigence trouvée dans les documents retenus. Précisez la norme ou élargissez les filtres.")
                st.write(f"3/3 · Comparaison avec {len(chunks)} extraits de normes")
                verdict = verifier_conformite(mesures, chunks, referentiel.strip(), contexte.strip())
                st.session_state["conf_resultat"] = {"verdict": verdict, "chunks": chunks}
                etat.update(label="Analyse terminée", state="complete", expanded=False)
            except Exception as e:
                etat.update(label="Analyse impossible", state="error")
                st.error(str(e)[:600])

    if st.session_state.get("conf_resultat"):
        st.markdown("---")
        afficher_conformite(st.session_state["conf_resultat"])


# --- Retours des utilisateurs (👍 / 👎 + commentaire) enregistrés dans la table « retours » ---
def _resume_sources(chunks: list) -> list:
    vus, sortie = set(), []
    for c in chunks or []:
        m = c.get("metadata", {})
        cle = (m.get("chemin") or m.get("fichier"), m.get("page"))
        if cle in vus:
            continue
        vus.add(cle)
        sortie.append({"fichier": m.get("fichier"), "chemin": m.get("chemin"), "page": m.get("page")})
        if len(sortie) >= 20:
            break
    return sortie


def enregistrer_retour(numero: int, question: str, reponse: str, chunks: list, note: int, commentaire: str = "") -> bool:
    try:
        supabase.table("retours").upsert({
            "utilisateur": st.session_state.get("username_courant", ""),
            "session": st.session_state.get("session_courante", ""),
            "numero_message": int(numero),
            "question_md5": hashlib.md5((question or "").encode("utf-8")).hexdigest(),
            "note": int(note),
            "question": (question or "")[:2000],
            "reponse": (reponse or "")[:6000],
            "commentaire": (commentaire or "")[:1000],
            "sources": _resume_sources(chunks),
            "version_app": globals().get("VERSION_APP", ""),
        }, on_conflict="utilisateur,session,numero_message,question_md5").execute()
        return True
    except Exception as e:
        st.session_state["retour_erreur"] = str(e)[:300]
        return False


def afficher_retour(cle: str, numero: int, question: str, reponse: str, chunks: list):
    """Pouces sous la réponse ; sur 👎, un champ de commentaire facultatif. Un vote modifié met à jour la ligne."""
    if not hasattr(st, "feedback"):
        return
    st.caption("Cette réponse vous a-t-elle aidé ?")
    valeur = st.feedback("thumbs", key=f"fb_{cle}")
    if valeur is None:
        return
    note = 1 if valeur == 1 else -1
    commentaire = ""
    if note == -1:
        commentaire = st.text_input("Qu'est-ce qui n'allait pas ? (facultatif, validez avec Entrée)", key=f"fbc_{cle}",
                                    placeholder="hors sujet, valeur erronée, source absente, trop long...")
    envoyes = st.session_state.setdefault("retours_envoyes", {})
    etat = {"note": note, "commentaire": commentaire.strip()}
    if envoyes.get(cle) != etat:
        if enregistrer_retour(numero, question, reponse, chunks, note, commentaire.strip()):
            envoyes[cle] = etat
            st.toast("Merci pour votre avis 🙏")
        else:
            st.caption("⚠️ Avis non enregistré (table « retours » absente ? exécutez le script SQL).")


@st.dialog("📊 Retours des utilisateurs", width="large")
def fenetre_retours():
    try:
        lignes = (supabase.table("retours")
                  .select("cree_le,utilisateur,note,question,commentaire,sources,reponse,version_app")
                  .order("cree_le", desc=True).limit(500).execute().data or [])
    except Exception as e:
        st.error("Table « retours » introuvable : exécutez le script SQL fourni (schema_retours_filtres.sql).")
        st.caption(str(e)[:200])
        return
    if not lignes:
        st.info("Aucun retour pour le moment.")
        return
    pos = sum(1 for l in lignes if l["note"] == 1)
    neg = len(lignes) - pos
    c1, c2, c3 = st.columns(3)
    c1.metric("Retours", len(lignes))
    c2.metric("👍 Utiles", pos)
    c3.metric("👎 À améliorer", neg)
    seulement_neg = st.checkbox("Afficher seulement les 👎", value=True)
    vue = [l for l in lignes if (l["note"] == -1 or not seulement_neg)]
    tableau = [{
        "Date": str(l["cree_le"])[:16].replace("T", " "),
        "Agent": l["utilisateur"],
        "Avis": "👍" if l["note"] == 1 else "👎",
        "Question": l["question"],
        "Commentaire": l["commentaire"] or "",
        "Sources": " ; ".join(f"{x.get('fichier')} p.{x.get('page')}" for x in (l.get("sources") or [])[:4]),
    } for l in vue]
    st.dataframe(tableau, use_container_width=True, hide_index=True)
    if vue:
        choix = st.selectbox("Voir la réponse donnée", range(len(vue)), format_func=lambda i: vue[i]["question"][:90])
        st.markdown(vue[choix]["reponse"] or "")
    import csv
    tampon = io.StringIO()
    ecrivain = csv.DictWriter(tampon, fieldnames=list(tableau[0].keys()) if tableau else ["Date"])
    ecrivain.writeheader()
    ecrivain.writerows(tableau)
    st.download_button("⬇️ Exporter en CSV", tampon.getvalue().encode("utf-8-sig"), "retours.csv", "text/csv")


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
            info_doc = analyser_nom_document(_chemin_norm(c)) if _chemin_norm(c) else None
            repere = f" · {info_doc['type']}" + (f" · {info_doc['annee']}" if info_doc and info_doc["annee"] else "") if info_doc else ""
            st.markdown(f"**Source {i} · {meta.get('fichier')}** — page {meta.get('page')} · {etiquette}{repere} (pertinence : {c['similarity']:.0%})")
            st.caption(nettoyer_remplissage(c["content"]))
            chemin_src = chemin_pour_telechargement(c)
            if peut_visualiser and chemin_src:
                if st.button("👁️ Visualiser cette source", key=f"vis_src_{prefixe_cle}_{i}"):
                    ouvrir_visionneuse(prefixe_cle, chemin_src, meta.get("fichier"), meta.get("page"), c["content"])
                bloc_lecture_vision(f"{prefixe_cle}_{i}", chemin_src, meta.get("page"))
            elif peut_visualiser:
                st.caption("⚠️ Ce document n'est pas retrouvé dans le dépôt Hugging Face (nom différent ou fichier retiré) : "
                           "visualisation et téléchargement impossibles.")
            st.divider()

    if nb_retirees:
        retires = {}
        for c in chunks:
            if _supprime(c):
                m = c["metadata"]
                retires[m.get("fichier") or "(nom inconnu)"] = m.get("chemin")
        st.warning("⚠️ " + str(nb_retirees) + " extrait(s) masqué(s) : le PDF n'existe pas sous le chemin enregistré dans "
                   "le dépôt Hugging Face — " + " ; ".join(f"**{n}** (chemin : {ch})" for n, ch in retires.items())
                   + ". Le fichier a été renommé, déplacé ou retiré : réindexez-le (barre latérale : « Tester un "
                     "fichier » puis « Synchroniser maintenant »).")

    try:
        for message in alertes_editions([c for c in chunks if not _supprime(c)]):
            st.warning(message)
    except Exception:
        pass  # l'alerte est un plus : elle ne doit jamais empêcher l'affichage des sources

    if not peut_visualiser:
        return

    fichiers_uniques = {}
    for c in chunks:
        if _supprime(c):
            continue
        meta = c["metadata"]
        chemin_doc = chemin_pour_telechargement(c)
        if chemin_doc:
            fichiers_uniques[chemin_doc] = meta.get("fichier") or chemin_doc.split("/")[-1]

    introuvables = {}
    for c in chunks:
        if not _supprime(c) and not chemin_pour_telechargement(c):
            m = c["metadata"]
            introuvables[m.get("fichier") or "(nom inconnu)"] = m.get("chemin")
    if introuvables:
        st.warning("⚠️ Document(s) cité(s) mais introuvable(s) dans le dépôt Hugging Face : "
                   + ", ".join(f"**{nom}**" for nom in introuvables)
                   + ". Le nom du PDF a changé depuis l'indexation, ou il a été retiré : "
                     "réindexez-le (barre latérale : « Tester un fichier » puis « Synchroniser maintenant »).")

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
VERSION_APP = "v2026.10.21-recherche-reference"
fichiers_autorises = None  # défini par les filtres de la barre latérale
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

    st.radio("Mode", [MODE_RECHERCHE, MODE_CONFORMITE], key="mode_app", label_visibility="collapsed")
    afficher_recherche_reference()
    fichiers_autorises = afficher_filtres()
    if st.session_state.get("document_cible"):
        fichiers_autorises = frozenset({st.session_state["document_cible"]})  # le document ciblé l'emporte
    elif fichiers_autorises is not None:
        st.caption(f"🎚️ Filtres actifs ({len(fichiers_autorises)} document(s))")

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
        if st.button("📊 Retours des utilisateurs", use_container_width=True, key="btn_retours"):
            fenetre_retours()
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


# --- Aiguillage : assistant de conformité ---
if st.session_state.get("mode_app") == MODE_CONFORMITE:
    page_conformite(fichiers_autorises)
    st.stop()

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

# --- Filtres (réglés dans la barre latérale) : rappel en haut de la conversation ---
question_vide = fichiers_autorises is not None and not fichiers_autorises
if question_vide:
    st.warning("Aucun document ne correspond aux filtres choisis : élargissez-les (barre latérale, « 🎚️ Filtres de "
               "recherche ») pour pouvoir poser une question.")
elif st.session_state.get("document_cible"):
    st.info(f"🎯 Vos questions portent uniquement sur **{st.session_state['document_cible'].split('/')[-1]}**. "
            "Pour revenir à toute la base, cliquez sur « ✖ Retirer cette limitation » dans la barre latérale.")
elif fichiers_autorises is not None:
    st.info(f"🎚️ Filtres actifs : la recherche porte sur {len(fichiers_autorises)} document(s). "
            "Modifiez-les dans la barre latérale (« 🎚️ Filtres de recherche »).")

for idx, echange in enumerate(historique):
    with st.chat_message("user" if echange["role"] == "user" else "assistant"):
        st.markdown(echange["content"])
        if echange["role"] == "assistant" and echange.get("sources"):
            _cle_msg = f"{st.session_state.session_courante}_msg{idx}"
            afficher_sources(echange["sources"], prefixe_cle=_cle_msg)
            afficher_retour(_cle_msg, idx, historique[idx - 1]["content"] if idx > 0 else "",
                            echange["content"], echange["sources"])

question = st.chat_input("Posez votre question ou enchaînez sur la précédente...", disabled=question_vide)

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
            chunks = rechercher_chunks(question_recherche, seulement_principales, fichiers_autorises)
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
                "Aucun passage pertinent trouvé dans les documents retenus par vos **filtres** (type / année). "
                "Élargissez les filtres pour chercher dans plus de documents."
                if fichiers_autorises is not None else
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
            _cle_msg = f"{st.session_state.session_courante}_msg{len(historique)}"
            afficher_sources(chunks, prefixe_cle=_cle_msg)
            st.caption(f"⏱️ Recherche : {t_recherche:.1f} s · Génération de la réponse : {t_generation:.1f} s")
            afficher_retour(_cle_msg, len(historique), question, reponse, chunks)
            historique.append({"role": "assistant", "content": reponse, "sources": chunks})
            sauvegarder_historique()
