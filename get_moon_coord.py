# pip install opencv-python numpy scipy
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates


# ---------- УТИЛИТЫ ----------
def circle_from_3pts(p1, p2, p3):
    # Возвращает (cx, cy, R) или None если точки почти коллинеарны
    (x1,y1), (x2,y2), (x3,y3) = p1, p2, p3
    a = x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2
    if abs(a) < 1e-6:
        return None
    b = ( (x1**2+y1**2)*(y3-y2) + (x2**2+y2**2)*(y1-y3) + (x3**2+y3**2)*(y2-y1) )
    c = ( (x1**2+y1**2)*(x2-x3) + (x2**2+y2**2)*(x3-x1) + (x3**2+y3**2)*(x1-x2) )
    cx = -b/(2*a)
    cy = -c/(2*a)
    R  = np.sqrt((cx-x1)**2 + (cy-y1)**2)
    return cx, cy, R

def kasa_init(points):
    # Быстрый «линейный» старт (algebraic LS), затем превратим в геом. МНК
    x = points[:,0]; y = points[:,1]
    x_m = x.mean(); y_m = y.mean()
    u = x - x_m; v = y - y_m
    Suu = np.sum(u*u); Svv = np.sum(v*v); Suv = np.sum(u*v)
    Suuu = np.sum(u*u*u); Svvv = np.sum(v*v*v)
    Suvv = np.sum(u*v*v); Svuu = np.sum(v*u*u)
    A = np.array([[Suu, Suv],[Suv, Svv]])
    b = 0.5*np.array([Suuu + Suvv, Svvv + Svuu])
    if np.linalg.cond(A) > 1e12:
        # fallback на простую оценку
        cx, cy = x_m, y_m
    else:
        uc, vc = np.linalg.solve(A, b)
        cx, cy = x_m + uc, y_m + vc
    R = np.mean(np.sqrt((x-cx)**2 + (y-cy)**2))
    return np.array([cx, cy, R], dtype=np.float64)

def geom_residuals(params, pts, weights=None):
    cx, cy, R = params
    r = np.sqrt((pts[:,0]-cx)**2 + (pts[:,1]-cy)**2) - R
    return r if weights is None else r * np.sqrt(np.clip(weights, 1e-6, None))

def ransac_circle(points, n_iter=2000, inlier_thr=1.5, min_inliers=50, rng=np.random.default_rng(0)):
    N = len(points)
    best_inliers = None
    best_model = None
    if N < 3: 
        return None, None
    idx = np.arange(N)
    for _ in range(n_iter):
        i1, i2, i3 = rng.choice(idx, size=3, replace=False)
        model = circle_from_3pts(points[i1], points[i2], points[i3])
        if model is None: 
            continue
        cx, cy, R = model
        d = np.abs(np.sqrt((points[:,0]-cx)**2 + (points[:,1]-cy)**2) - R)
        inliers = np.where(d < inlier_thr)[0]
        if best_inliers is None or len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = (cx, cy, R)
    if best_inliers is None or len(best_inliers) < min_inliers:
        return None, None
    return best_model, best_inliers

def subpixel_refine_along_normal(gray_f32, pts, gx, gy, radius=2.0, step=0.25):
    refined = []
    h, w = gray_f32.shape
    for (x,y, Gx, Gy) in zip(pts[:,0], pts[:,1], gx, gy):
        gnorm = float(np.hypot(Gx, Gy))
        if gnorm < 1e-6:
            refined.append([x, y])
            continue

        nx, ny = Gx/gnorm, Gy/gnorm
        ts = np.arange(-radius, radius + 1e-9, step, dtype=np.float32)
        xs = x + ts*nx
        ys = y + ts*ny

        # ВНИМАНИЕ: порядок координат — (row=y, col=x)
        I = map_coordinates(
            gray_f32, np.vstack([ys, xs]),
            order=1, mode='nearest'
        ).astype(np.float32)

        if I.size < 2:
            refined.append([x, y])
            continue

        Imin, Imax = float(I.min()), float(I.max())
        half = 0.5*(Imin + Imax)

        dI = np.diff(I)
        cross = (I[:-1]-half)*(I[1:]-half) <= 0
        if not np.any(cross):
            refined.append([x, y])
            continue

        k = int(np.argmax(np.abs(dI)))
        if I[k] == I[k+1]:
            t_sub = ts[k]
        else:
            t_sub = ts[k] + (half - I[k]) * (ts[k+1]-ts[k]) / (I[k+1]-I[k])

        xr = float(np.clip(x + t_sub*nx, 0, w-1))
        yr = float(np.clip(y + t_sub*ny, 0, h-1))
        refined.append([xr, yr])

    return np.array(refined, dtype=np.float32)


# ---------- ОСНОВНАЯ ПРОЦЕДУРА ----------
def fit_moon_circle(image_path, undistort=None, visualize=True):
    """
    image_path: путь к изображению
    undistort: (camera_matrix, dist_coeffs) если есть калибровка OpenCV — сначала раздисторсим
    visualize: рисовать ли результат
    return: (cx, cy, R), inliers (Nx2)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)
    # 1) (опционально) Undistort
    if undistort is not None:
        K, d = undistort
        img = cv2.undistort(img, K, d)
    # 2) Контраст/шум
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(img)
    g = cv2.GaussianBlur(g, (0,0), sigmaX=1.0)
    g32 = g.astype(np.float32)

    # 3) Градиенты и Canny (порог по медиане)
    v = np.median(g)
    low = int(max(0, 0.66*v))
    high = int(min(255, 1.33*v))
    edges = cv2.Canny(g, threshold1=low, threshold2=high, L2gradient=True)
    # 4) Контуры — берём несколько самых длинных
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key=lambda c: len(c), reverse=True)[:10]
    pts = np.vstack([c.reshape(-1,2) for c in cnts]) if cnts else np.empty((0,2), np.int32)
    if len(pts) < 10:
        raise RuntimeError("Слишком мало точек кромки. Проверьте пороги/предобработку.")

    pts = pts.astype(np.float32)

    # 5) Субпиксель: нормаль = направление градиента (Sobel)
    Gx = cv2.Sobel(g32, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(g32, cv2.CV_32F, 0, 1, ksize=3)
    gx = Gx[pts[:,1].astype(int), pts[:,0].astype(int)]
    gy = Gy[pts[:,1].astype(int), pts[:,0].astype(int)]
    pts_ref = subpixel_refine_along_normal(g32, pts, gx, gy, radius=2.0, step=0.25)

    # 6) RANSAC для устойчивых инлайеров
    model0, inliers_idx = ransac_circle(pts_ref, n_iter=3000, inlier_thr=1.5, min_inliers=80)
    if model0 is None:
        # запасной план: инициализация Kasa и без RANSAC
        model0 = kasa_init(pts_ref)
        inliers = pts_ref
    else:
        inliers = pts_ref[inliers_idx]

    # 7) Геометрическая робастная донастройка (ортогональные расстояния)
    # веса = модуль градиента (сильные кромки чуть важнее)
    mgrad = np.hypot(
        Gx[inliers[:,1].astype(int), inliers[:,0].astype(int)],
        Gy[inliers[:,1].astype(int), inliers[:,0].astype(int)]
    )
    x0 = np.array(model0, dtype=np.float64)
    res = least_squares(geom_residuals, x0, args=(inliers, mgrad),
                        loss='soft_l1', f_scale=1.0, max_nfev=500)
    cx, cy, R = res.x

    if visualize:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for p in inliers.astype(int):
            cv2.circle(out, tuple(p), 1, (0,255,0), -1)
        cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(R)), (255,0,0), 2)
        cv2.circle(out, (int(round(cx)), int(round(cy))), 3, (0,0,255), -1)
        cv2.putText(out, f"cx={cx:.2f}, cy={cy:.2f}, R={R:.2f}", (10*3,30*3),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0,200,255), 4, cv2.LINE_AA)
        cv2.imshow("Moon circle fit", cv2.resize(out, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return (float(cx), float(cy), float(R)), inliers


#fit_moon_circle(r"D:\Avocation\Sky\80. 07-08.09.2025\stack\jpg\DSC07984.JPG")
fit_moon_circle(r"D:\Avocation\Sky\80. 07-08.09.2025\jpg\DSC08037.JPG")
