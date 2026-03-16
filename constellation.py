"""
COSMIC SIMULATION ENGINE  v8.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hands together  →  BLACK HOLE   (particles spiral in)
Hands apart     →  SUPERNOVA    (particles explode out, once per spread)
Otherwise       →  STELLAR DRIFT (particles hug your body)

Particles always live on your skeleton — no spacebar needed.

Controls:  M = debug overlay   S = screenshot   ESC = quit
"""

import sys, os, math, time
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import pygame.gfxdraw
import cv2
import mediapipe as mp

# ───────────────────────────────────────────────
#  SCREEN / PARTICLE CONSTANTS
# ───────────────────────────────────────────────
WIDTH, HEIGHT = 1280, 720
TARGET_FPS    = 60
MAX_PARTICLES = 2000

_HERE         = os.path.dirname(os.path.abspath(__file__))
BG_IMAGE_PATH = os.path.join(_HERE, "background.png")

MODE_DRIFT     = "STELLAR DRIFT"
MODE_BLACKHOLE = "BLACK HOLE"
MODE_SUPERNOVA = "SUPERNOVA"

BH_ENTER          = 175
BH_EXIT           = 280
BH_CONFIRM_FRAMES = 3
BH_RELEASE_FRAMES = 5

SN_ENTER          = 620
SN_EXIT           = 530
SN_COOLDOWN       = 3.0

def clamp(v, lo, hi): return max(lo, min(hi, v))
def lerp(a, b, t):    return a + (b - a) * t
def lerp_col(a, b, t): return tuple(int(lerp(a[i], b[i], t)) for i in range(3))


# ───────────────────────────────────────────────
#  SOUND MANAGER
# ───────────────────────────────────────────────
class SoundManager:
    SR = 44100

    def __init__(self):
        self._ok = False
        try:
            pygame.mixer.pre_init(self.SR, -16, 2, 256)
            pygame.mixer.init()
            pygame.mixer.set_num_channels(16)
            self._sounds = {}
            self._build()
            self._ok = True
            print("  [Audio] OK")
        except Exception as e:
            print(f"  [Audio] failed: {e}")

    def _t(self, d): return np.linspace(0, d, int(self.SR * d), endpoint=False)

    def _env(self, a, atk=0.02, rel=0.08):
        n = len(a)
        e = np.ones(n)
        ak = min(int(self.SR*atk), n//4)
        rk = min(int(self.SR*rel), n//4)
        if ak: e[:ak]  = np.linspace(0,1,ak)
        if rk: e[-rk:] = np.linspace(1,0,rk)
        return a * e

    def _make(self, a):
        a  = np.clip(a,-1,1)
        i16 = (a*28000).astype(np.int16)
        return pygame.sndarray.make_sound(np.column_stack([i16,i16]))

    def _build(self):
        # ambient hum
        t = self._t(2.0)
        d = np.sin(2*np.pi*55*t)*.5 + np.sin(2*np.pi*110*t)*.25 + np.sin(2*np.pi*82.5*t)*.2
        self._sounds['ambient'] = self._make(self._env(d,.4,.4)*.3)

        # black hole: deep rumble
        t = self._t(1.4)
        f = 80*np.exp(-t*.8)
        b = np.sin(2*np.pi*np.cumsum(f)/self.SR)
        b += np.sin(2*np.pi*40*t)*.4 + np.random.randn(len(t))*.04
        self._sounds['blackhole'] = self._make(self._env(b,.05,.4)*.65)

        # supernova: crack + boom
        t  = self._t(0.6)
        sn = (np.random.randn(len(t))*np.exp(-t*12)*.6
              + np.sin(2*np.pi*400*t)*np.exp(-t*8)*.3
              + (np.sin(2*np.pi*60*t)+np.sin(2*np.pi*90*t))*np.exp(-t*5)*.5)
        self._sounds['supernova'] = self._make(self._env(sn,.001,.15)*.85)

        ch = pygame.mixer.Channel(0)
        ch.set_volume(0.18)
        ch.play(self._sounds['ambient'], loops=-1)

    def play(self, name, vol=0.7):
        if not self._ok: return
        s = self._sounds.get(name)
        if not s: return
        ch = pygame.mixer.find_channel(True)
        if ch:
            ch.set_volume(vol)
            ch.play(s)


# ───────────────────────────────────────────────
#  GESTURE RECOGNISER  v8
# ───────────────────────────────────────────────
class GestureRecogniser:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self._slw = None
        self._srw = None
        self._K   = 0.50

        self._bh_in    = 0
        self._bh_out   = 0
        self._bh_on    = False

        self._sn_cd    = 0.0
        self._sn_armed = True

        self._dbg = 0

    def _wpx(self, lm, idx):
        l = lm[idx]
        if getattr(l,'visibility',1.0) < 0.05: return None
        if getattr(l,'presence',  1.0) < 0.05: return None
        return np.array([l.x * self.w, l.y * self.h], dtype=np.float64)

    def update(self, landmarks, dt):
        self._sn_cd = max(0.0, self._sn_cd - dt)

        if landmarks is None:
            self._slw = self._srw = None
            self._bh_in = self._bh_out = 0
            self._bh_on = False
            self._sn_armed = True
            return MODE_DRIFT, None, False

        lm = landmarks
        lr = self._wpx(lm, 15)
        rr = self._wpx(lm, 16)

        l = lr if lr is not None else self._slw
        r = rr if rr is not None else self._srw
        if l is None or r is None:
            self._bh_in = 0
            return MODE_DRIFT, None, False

        K = self._K
        if self._slw is None:
            self._slw, self._srw = l.copy(), r.copy()
        else:
            self._slw = self._slw*(1-K) + l*K
            self._srw = self._srw*(1-K) + r*K

        d_raw    = float(np.linalg.norm(l - r))
        d_smooth = float(np.linalg.norm(self._slw - self._srw))
        d_enter  = min(d_raw, d_smooth)
        d_exit   = max(d_raw, d_smooth)
        d_sn     = d_smooth

        self._dbg += 1
        if self._dbg % 25 == 0:
            st = "BH" if self._bh_on else "--"
            print(f"  [G] in={d_enter:.0f} out={d_exit:.0f} sn={d_sn:.0f}  "
                  f"bh_in={self._bh_in}/{BH_CONFIRM_FRAMES}  "
                  f"bh_out={self._bh_out}/{BH_RELEASE_FRAMES}  [{st}]")

        sn_fired = False

        if not self._bh_on:
            if d_enter < BH_ENTER:
                self._bh_in  += 1
                self._bh_out  = 0
            else:
                self._bh_in   = max(0, self._bh_in - 1)
            if self._bh_in >= BH_CONFIRM_FRAMES:
                self._bh_on  = True
                self._bh_in  = 0
                self._bh_out = 0
                print(f"  [BH] ON   d={d_enter:.0f}px")
        else:
            self._bh_in = 0
            if d_exit > BH_EXIT:
                self._bh_out += 1
            else:
                self._bh_out  = max(0, self._bh_out - 1)
            if self._bh_out >= BH_RELEASE_FRAMES:
                self._bh_on   = False
                self._bh_out  = 0
                print(f"  [BH] OFF  d={d_exit:.0f}px")

        if self._bh_on:
            bh_pos = (l + r) / 2.0
            return MODE_BLACKHOLE, bh_pos, False

        if d_sn > SN_ENTER:
            if self._sn_armed and self._sn_cd <= 0:
                sn_fired      = True
                self._sn_cd   = SN_COOLDOWN
                self._sn_armed = False
                print(f"  [SN] FIRE  d={d_sn:.0f}px")
            return (MODE_SUPERNOVA if sn_fired else MODE_DRIFT), None, sn_fired

        if d_sn < SN_EXIT:
            self._sn_armed = True

        return MODE_DRIFT, None, False

    @property
    def live_dist(self):
        if self._slw is None: return 0.0
        return float(np.linalg.norm(self._slw - self._srw))


# ───────────────────────────────────────────────
#  ACCRETION DISK  — now dark / black
# ───────────────────────────────────────────────
class AccretionDisk:
    def __init__(self):
        self.alpha  = 0.0
        self.center = np.array([WIDTH/2.0, HEIGHT/2.0])

    def update(self, on, center, dt):
        self.alpha = clamp(self.alpha + ((1.0 if on else 0.0) - self.alpha)*dt*30, 0, 1)
        if on: self.center = center.copy()

    def draw(self, surf):
        if self.alpha < 0.01: return
        cx, cy = int(self.center[0]), int(self.center[1])
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        a = self.alpha

        # Dark elliptical rings — deep black/dark-gray tones
        for r, th, col in [
            (140, 24, ( 20,  20,  30)),
            (108, 16, ( 40,  40,  55)),
            ( 76, 11, ( 60,  60,  80)),
            ( 48,  7, ( 80,  80, 100)),
        ]:
            pygame.draw.ellipse(s, (*col, int(clamp(210*a, 0, 255))),
                                (cx-r, cy-r//3, r*2, r*2//3), th)

        # Pure black void core
        pygame.draw.circle(s, (  0,   0,   0, int(255*a)), (cx, cy), 32)

        # Faint dark-purple event horizon rim
        pygame.draw.circle(s, ( 80,  80, 120, int(140*a)), (cx, cy), 40, 2)

        # Near-black inner glow — almost invisible
        for gr, ga in [(22, 80), (16, 120), (10, 180)]:
            pygame.draw.circle(s, ( 10,  10,  15, int(ga*a)), (cx, cy), gr)

        surf.blit(s, (0, 0))


# ───────────────────────────────────────────────
#  SHOCKWAVE
# ───────────────────────────────────────────────
class ShockwaveManager:
    def __init__(self): self.waves = []

    def trigger(self, cx, cy):
        self.waves += [
            {'cx':cx,'cy':cy,'r':4,  'maxr':1100,'alpha':255,'col':(255,210, 80),'spd':950},
            {'cx':cx,'cy':cy,'r':2,  'maxr': 850,'alpha':200,'col':(120,180,255),'spd':750},
            {'cx':cx,'cy':cy,'r':1,  'maxr': 600,'alpha':160,'col':(255,255,255),'spd':600},
        ]

    def update(self, dt):
        for w in self.waves:
            w['r']    += dt * w['spd']
            w['alpha'] = max(0, w['alpha'] - dt*300)
        self.waves = [w for w in self.waves if w['alpha'] > 0]

    def draw(self, surf):
        if not self.waves: return
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for w in self.waves:
            a  = int(clamp(w['alpha'],0,255))
            r  = int(w['r'])
            if r <= 0: continue
            th = max(1, int(5*(1-w['r']/w['maxr']))+1)
            pygame.draw.circle(s,(*w['col'],a),(int(w['cx']),int(w['cy'])),r,th)
        surf.blit(s,(0,0))


# ───────────────────────────────────────────────
#  HUD
# ───────────────────────────────────────────────
class HUD:
    C_CYAN  = (100,220,255)
    C_GOLD  = (255,200, 60)
    C_RED   = (255, 80, 70)
    C_WHITE = (220,228,255)
    C_PANEL = ( 8, 14, 36, 160)

    def __init__(self, w, h):
        self.w, self.h      = w, h
        self._t             = 0.0
        self.mode           = MODE_DRIFT
        self.fps            = 60
        self.n_particles    = MAX_PARTICLES
        self.person         = False
        self.wrist_px       = 0.0
        self.flash_msg      = ""
        self.flash_timer    = 0.0
        self.flash_col      = self.C_CYAN
        self.screenshot_msg = 0.0

        pygame.font.init()
        self._ff = pygame.font.SysFont("Courier New", 50, bold=True)
        self._fm = pygame.font.SysFont("Courier New", 19, bold=True)
        self._fs = pygame.font.SysFont("Courier New", 12)
        self._ft = pygame.font.SysFont("Courier New", 10)

    def update(self, dt, mode, fps, n, person, wpx):
        self._t          += dt
        self.mode         = mode
        self.fps          = fps
        self.n_particles  = n
        self.person       = person
        self.wrist_px     = wpx
        if self.flash_timer    > 0: self.flash_timer    -= dt
        if self.screenshot_msg > 0: self.screenshot_msg -= dt

    def flash(self, msg, col, dur=2.0):
        self.flash_msg   = msg
        self.flash_col   = col
        self.flash_timer = dur

    def _mc(self):
        return {MODE_BLACKHOLE:self.C_GOLD, MODE_SUPERNOVA:self.C_RED}.get(self.mode, self.C_CYAN)

    def _panel(self, ov, x, y, w, h):
        s = pygame.Surface((w,h), pygame.SRCALPHA)
        s.fill(self.C_PANEL)
        ov.blit(s,(x,y))
        mc = self._mc()
        pygame.draw.rect(ov,(*mc,50),(x,y,w,h),1)

    def draw(self, surf):
        ov    = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        mc    = self._mc()
        pulse = 0.72 + 0.28*abs(math.sin(self._t*2.5))

        L = 30
        for (cx,cy),(dx,dy) in [
            ((8,8),(1,1)),((self.w-8,8),(-1,1)),
            ((8,self.h-8),(1,-1)),((self.w-8,self.h-8),(-1,-1))
        ]:
            ca = int(200*pulse)
            pygame.draw.line(ov,(*mc,ca),(cx,cy),(cx+dx*L,cy),2)
            pygame.draw.line(ov,(*mc,ca),(cx,cy),(cx,cy+dy*L),2)
            pygame.draw.circle(ov,(*mc,ca),(cx,cy),2)

        self._panel(ov, 8, 8, 300, 78)

        dc = (60,220,80) if self.person else (180,60,60)
        pygame.draw.circle(ov,(*dc,220),(22,22),5)
        rr = int(8+3*abs(math.sin(self._t*4)))
        pygame.draw.circle(ov,(*dc,int(75*pulse)),(22,22),rr,1)
        ov.blit(self._fs.render(
            "ENTITY LOCKED" if self.person else "SCANNING...",
            True,(*dc,210)),(34,15))
        ov.blit(self._ft.render(
            f"FPS {int(self.fps):>3}   PARTICLES {self.n_particles:>4}",
            True,(*self.C_WHITE,120)),(14,34))

        if self.person and self.wrist_px > 0:
            BW, BX, BY, BH2 = 274, 14, 50, 10
            MV = SN_ENTER + 100
            fi = int(BW * min(1.0, self.wrist_px / MV))

            if   self.wrist_px < BH_ENTER: bc = self.C_GOLD
            elif self.wrist_px > SN_ENTER: bc = self.C_RED
            else:                          bc = self.C_CYAN

            pygame.draw.rect(ov,(20,26,55,200),(BX,BY,BW,BH2))
            if fi > 0:
                pygame.draw.rect(ov,(*bc,200),(BX,BY,fi,BH2))

            for val, col in [(BH_ENTER,self.C_GOLD),(BH_EXIT,self.C_GOLD),
                             (SN_EXIT,self.C_RED),(SN_ENTER,self.C_RED)]:
                mx = BX + int(BW * val / MV)
                pygame.draw.line(ov,(*col,160),(mx,BY-2),(mx,BY+BH2+1),1)

            ov.blit(self._ft.render(
                f"{int(self.wrist_px)}px   BH<{BH_ENTER}  SN>{SN_ENTER}",
                True,(*bc,190)),(BX, BY+BH2+3))

        badge = self._fm.render(self.mode, True, (*mc,int(230*pulse)))
        bw    = badge.get_width()
        bx    = self.w - bw - 16
        self._panel(ov, bx-8, 6, bw+16, 42)
        ov.blit(badge,(bx,11))
        bf = int(bw*abs(math.sin(self._t*1.5)))
        pygame.draw.rect(ov,(*mc,45),(bx,36,bw,2))
        pygame.draw.rect(ov,(*mc,210),(bx,36,bf,2))

        tips = [
            (f"Hands together (<{BH_ENTER}px)", "→ Black Hole"),
            (f"Hands apart    (>{SN_ENTER}px)", "→ Supernova"),
        ]
        gh = 13*len(tips)+14
        gy = self.h - gh - 24
        self._panel(ov, 8, gy, 272, gh)
        for i,(trigger,effect) in enumerate(tips):
            active = (("Black" in effect and "BLACK" in self.mode)
                      or ("Nova"  in effect and "NOVA"  in self.mode))
            c2 = (*mc,205) if active else (110,130,160,155)
            ov.blit(self._ft.render(f"{trigger}  {effect}", True, c2),(15, gy+7+i*13))

        ctrl = "M Debug  |  S Screenshot  |  ESC Quit"
        ov.blit(self._ft.render(ctrl,True,(65,80,105,120)),
                (self.w//2 - self._ft.size(ctrl)[0]//2, self.h-13))

        if self.flash_timer > 0:
            al  = int(255 * min(1.0, self.flash_timer/0.45))
            fc  = self.flash_col
            for ox,oy in [(3,3),(-3,3),(3,-3),(-3,-3)]:
                g = self._ff.render(self.flash_msg,True,(*fc,clamp(al//6,0,255)))
                ov.blit(g,(self.w//2-g.get_width()//2+ox,
                           self.h//2-g.get_height()//2-55+oy))
            fs = self._ff.render(self.flash_msg,True,(*fc,clamp(al,0,255)))
            ov.blit(fs,(self.w//2-fs.get_width()//2, self.h//2-fs.get_height()//2-55))
            tw = fs.get_width()
            tx = self.w//2 - tw//2
            ty = self.h//2 - 55 + fs.get_height() + 3
            lf = int(tw * min(1.0,(2.0-self.flash_timer)/0.35))
            pygame.draw.rect(ov,(*fc,clamp(al//2,0,255)),(tx,ty,tw,2))
            pygame.draw.rect(ov,(*fc,clamp(al,    0,255)),(tx,ty,lf,2))

        if self.screenshot_msg > 0:
            a2 = int(clamp(self.screenshot_msg*130,0,255))
            self._panel(ov,self.w//2-130,58,260,34)
            ss = self._fm.render("★ SCREENSHOT SAVED ★",True,(*self.C_GOLD,a2))
            ov.blit(ss,(self.w//2-ss.get_width()//2,65))

        surf.blit(ov,(0,0))


# ───────────────────────────────────────────────
#  HELPERS
# ───────────────────────────────────────────────
BODY_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(25,27),
    (24,26),(26,28),(27,31),(28,32),
    (3,7),(6,8),(9,10),(1,2),(0,4),
]
NUM_INTERP = 8

def build_targets(lms):
    valid = {}
    pts   = []
    for i, lm in enumerate(lms):
        if getattr(lm,'visibility',1.0)>0.3 and getattr(lm,'presence',1.0)>0.3:
            p = np.array([lm.x*WIDTH, lm.y*HEIGHT])
            valid[i] = p
            pts.append(p)
    for si,ei in BODY_CONNECTIONS:
        if si in valid and ei in valid:
            p1,p2 = valid[si], valid[ei]
            for k in range(1, NUM_INTERP):
                f = k/NUM_INTERP
                pts.append(p1*(1-f)+p2*f)
    return np.array(pts) if pts else np.array([])


# ───────────────────────────────────────────────
#  MAIN
# ───────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("COSMIC SIMULATION ENGINE  v8.0")
    clock  = pygame.time.Clock()

    print("=" * 58)
    print("  COSMIC SIMULATION ENGINE  v8.0")
    print(f"  BH enter<{BH_ENTER}px  exit>{BH_EXIT}px  "
          f"({BH_CONFIRM_FRAMES}in/{BH_RELEASE_FRAMES}out frames)")
    print(f"  SN enter>{SN_ENTER}px  exit<{SN_EXIT}px  cooldown={SN_COOLDOWN}s")
    print("=" * 58)

    print("  Loading assets ...")
    try:
        bg = pygame.transform.scale(
            pygame.image.load(BG_IMAGE_PATH).convert(),(WIDTH,HEIGHT))
    except Exception as e:
        print(f"  (background fallback: {e})")
        bg = pygame.Surface((WIDTH,HEIGHT)); bg.fill((2,6,22))

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        print("  ERROR: no webcam"); sys.exit(1)

    model_path = os.path.join(_HERE, "pose_landmarker.task")
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.50,
        min_pose_presence_confidence=0.50,
        min_tracking_confidence=0.50,
    )
    pose = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    gesture    = GestureRecogniser(WIDTH, HEIGHT)
    accretion  = AccretionDisk()
    shockwaves = ShockwaveManager()
    hud        = HUD(WIDTH, HEIGHT)
    sound      = SoundManager()

    num_p  = MAX_PARTICLES
    pos    = np.random.rand(num_p,2) * [WIDTH,HEIGHT]
    vel    = np.zeros((num_p,2))
    active = np.ones(num_p, dtype=bool)
    colors = np.full((num_p,3),[0.,200.,255.])

    mode       = MODE_DRIFT
    last_mode  = mode
    person     = False
    targets    = np.array([])
    tgt_col    = [0., 200., 255.]
    bh_center  = None
    debug_mode = False
    flash_white= 0.0
    prev_t     = time.time()
    running    = True

    trail_surf = pygame.Surface((WIDTH,HEIGHT), pygame.SRCALPHA)

    BH_G    = 180000.0
    BH_DAMP = 0.88
    BH_CORE = 28
    BH_SWRL = 0.18
    BH_CAP  = 900.0

    print("  READY\n")

    while running:
        now   = time.time()
        dt    = clamp(now - prev_t, 0.001, 0.05)
        prev_t= now
        clock.tick(TARGET_FPS)
        fps = clock.get_fps()

        if fps < 26 and num_p > 400:
            cut   = min(40, num_p-400)
            num_p-= cut
            pos=pos[:num_p]; vel=vel[:num_p]
            active=active[:num_p]; colors=colors[:num_p]

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_m:
                    debug_mode = not debug_mode
                elif ev.key == pygame.K_s:
                    fn = f"cosmic_{int(now)}.png"
                    pygame.image.save(screen, fn)
                    hud.screenshot_msg = 2.5
                    print(f"  Screenshot → {fn}")

        ret, frame = cap.read()
        if not ret: continue
        frame   = cv2.flip(frame, 1)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts_ms   = int(now * 1000)
        results = pose.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)

        bh_center     = None
        supernova_now = False
        targets       = np.array([])

        if results.pose_landmarks:
            if not person:
                person = True
                hud.flash("ENTITY DETECTED", (60,220,80), 2.2)
                print("  Entity detected.")

            lms = results.pose_landmarks[0]
            mode, bh_center, supernova_now = gesture.update(lms, dt)
            targets = build_targets(lms)

            try:
                lw,rw = lms[15], lms[16]
                if getattr(lw,'visibility',0)>0.3 and getattr(rw,'visibility',0)>0.3:
                    ny = clamp((lw.y+rw.y)/2, 0, 1)
                    tgt_col = [int(lerp(0,200,ny)), int(lerp(200,80,ny)), 255]
            except Exception:
                pass

            active[:] = True

        else:
            mode = MODE_DRIFT
            gesture.update(None, dt)
            if person:
                person = False

        if mode != last_mode:
            prev_mode = last_mode
            last_mode = mode

            if mode == MODE_BLACKHOLE:
                sound.play('blackhole', 0.7)
                hud.flash("BLACK HOLE", (255,200,60), 2.0)

        if supernova_now:
            sound.play('supernova', 0.9)
            shockwaves.trigger(WIDTH//2, HEIGHT//2)
            flash_white = 1.0
            hud.flash("SUPERNOVA!", (255,80,70), 2.2)

        if mode == MODE_BLACKHOLE and len(targets) > 0:
            n_dead = int(np.sum(~active))
            if n_dead > 0:
                idx = np.where(~active)[0]
                tidx = np.random.randint(0, len(targets), n_dead)
                pos[idx]    = targets[tidx] + np.random.randn(n_dead,2)*18
                pos[idx,0]  = np.clip(pos[idx,0], 0, WIDTH-1)
                pos[idx,1]  = np.clip(pos[idx,1], 0, HEIGHT-1)
                vel[idx]    = np.random.randn(n_dead,2)*1.5
                active[idx] = True
        elif mode != MODE_BLACKHOLE and len(targets) > 0:
            outliers = np.where(
                (pos[:,0]<0)|(pos[:,0]>=WIDTH)|(pos[:,1]<0)|(pos[:,1]>=HEIGHT)
            )[0]
            if len(outliers) > 0:
                tidx = np.random.randint(0, len(targets), len(outliers))
                pos[outliers]    = targets[tidx] + np.random.randn(len(outliers),2)*25
                vel[outliers]    = 0
                active[outliers] = True

        accretion.update(
            mode == MODE_BLACKHOLE,
            bh_center if bh_center is not None else np.array([WIDTH/2.,HEIGHT/2.]),
            dt)
        shockwaves.update(dt)
        hud.update(dt, mode, fps, int(np.sum(active)), person, gesture.live_dist)

        G      = 5000.0
        DAMP   = 0.95
        AURA   = 40.0
        pt     = targets.copy() if len(targets)>0 else np.array([])

        if supernova_now or mode == MODE_SUPERNOVA:
            G, DAMP = -34000.0, 0.98

        if mode == MODE_BLACKHOLE and bh_center is not None:
            ap = pos[active]; av = vel[active]
            bh = bh_center.astype(np.float64)
            d  = bh - ap
            dsq= d[:,0]**2 + d[:,1]**2 + 1.0
            dn = np.sqrt(dsq)
            fm = BH_G / dsq
            nm = d / dn[:,np.newaxis]
            tg = np.stack([-d[:,1],d[:,0]],axis=1)
            tg/= np.linalg.norm(tg,axis=1,keepdims=True)+1e-9
            acc= nm*fm[:,np.newaxis] + tg*(fm*BH_SWRL)[:,np.newaxis]
            av = av*BH_DAMP + acc*dt
            sp = np.linalg.norm(av,axis=1,keepdims=True)
            av = np.where(sp>BH_CAP, av/sp*BH_CAP, av)
            ap+= av*dt*TARGET_FPS
            pos[active]=ap; vel[active]=av

            aidx = np.where(active)[0]
            dsts = np.linalg.norm(pos[aidx]-bh_center,axis=1)
            done = aidx[dsts < BH_CORE]
            if len(done):
                active[done]=False; vel[done]=0

        elif len(pt)>0 and np.any(active):
            ap = pos[active]; av = vel[active]
            if len(pt)==1:
                df  = pt[0]-ap
                dsq = np.sum(df**2,axis=1)+1.0
                acc = (df/np.sqrt(dsq)[:,np.newaxis]) * (G/dsq)[:,np.newaxis]
            else:
                da  = pt[np.newaxis,:,:] - ap[:,np.newaxis,:]
                dsa = np.sum(da**2,axis=2)
                ci  = np.argmin(dsa,axis=1)
                cd  = da[np.arange(len(ap)),ci]
                cds = dsa[np.arange(len(ap)),ci]+20.0
                cdd = np.sqrt(cds)
                if supernova_now or mode==MODE_SUPERNOVA:
                    fm = G/cds
                else:
                    err= cdd - AURA
                    fm = np.clip(0.15*err,-10.,10.) + np.random.randn(len(ap))*0.5
                acc = (cd/cdd[:,np.newaxis]) * fm[:,np.newaxis]
            av+=acc; av*=DAMP; ap+=av
            pos[active]=ap; vel[active]=av

        colors = colors*0.88 + np.array(tgt_col,dtype=float)*0.12

        screen.blit(bg,(0,0))

        dark = pygame.Surface((WIDTH,HEIGHT),pygame.SRCALPHA)
        dark.fill((0,0,12,100))
        screen.blit(dark,(0,0))

        trail_surf.fill((0,0,0,16))
        screen.blit(trail_surf,(0,0))

        ai = np.where(active)[0]
        for i in ai:
            x,y = int(pos[i,0]), int(pos[i,1])
            if not (0<=x<WIDTH and 0<=y<HEIGHT): continue
            sp  = float(np.linalg.norm(vel[i]))
            sz  = 3 if sp>10 else 2
            c   = (clamp(int(colors[i,0]),0,255),
                   clamp(int(colors[i,1]),0,255),
                   clamp(int(colors[i,2]),0,255))
            try:
                pygame.gfxdraw.aacircle(screen,x,y,sz,c)
                pygame.gfxdraw.filled_circle(screen,x,y,sz,c)
            except (ValueError,OverflowError):
                pass
            if sp>14 and mode==MODE_SUPERNOVA:
                vn = vel[i]/(sp+1e-6)*min(sp*1.1,18)
                pygame.draw.line(screen,(*c,80),(x,y),(int(x-vn[0]),int(y-vn[1])),1)

        accretion.draw(screen)
        shockwaves.draw(screen)

        if flash_white > 0:
            fw = pygame.Surface((WIDTH,HEIGHT))
            fw.fill((255,255,255))
            fw.set_alpha(int(255*flash_white))
            screen.blit(fw,(0,0))
            flash_white = max(0., flash_white - dt*5.5)

        if debug_mode and len(targets)>0:
            for tx,ty in targets:
                pygame.draw.circle(screen,(255,0,0),(int(tx),int(ty)),3)

        hud.draw(screen)
        pygame.display.flip()

    cap.release()
    try: pose.close()
    except Exception: pass
    pygame.quit()
    print("  Session ended.")


if __name__ == "__main__":
    main()