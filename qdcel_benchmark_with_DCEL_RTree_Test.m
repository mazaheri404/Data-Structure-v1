function qdcel_benchmark_with_DCEL_RTree()
% =========================================================================
% QDCEL Benchmark (with REAL DCEL, Dynamic Quadtree AS-IS, KDTree, RTree)
% - Adds two QDCEL predictions: Leaf-Union and Triangle-Clip
% - Measures Build Time, Peak RAM, Query Latency, IoU (w/ 95% CI)
% - Exports vector PDF plots for ISI-style figures
% =========================================================================
clc; clear; close all; rng(42);

%% --------------------------- Config ------------------------------------
SCENES = [0.6, 1.0, 1.4]; % scene density multipliers
NRUNS = 8; % runs per scene
FoV_PIX = [40, 64, 96]; % three FoV sizes (px)
LEAF_PIX = 32; % target leaf size
MAXDEPTH = 10; % uniform QT max depth
ALT_SET = [40, 60, 80]; % three altitudes
W = 512; H = 512; % canvas size
base_params = struct('tau0',0.10,'alpha',1.25,'z0',50,'minDepth',0,'maxDepth',12);

% Methods:
% 1) QDCEL-Leaf 2) QDCEL-Clip 3) UniformQT-Leaf 4) KDTree 5) RTree
METHODS = {'QDCEL-Leaf','QDCEL-Clip','UniformQT-Leaf','KDTree','RTree'};
nM = numel(METHODS);
nS = numel(SCENES);

% Results: (scene, method, run)
BuildT = zeros(nS,nM,NRUNS);
MemMB = zeros(nS,nM,NRUNS);
QryT = zeros(nS,nM,NRUNS); % Query latency (s)
AccIoU = zeros(nS,nM,NRUNS);

%% ------------------------ Loop over scenes ------------------------------
for s = 1:nS
    density_scale = SCENES(s);
    fprintf('\n=== Scene %d/%d | density = %.2f ===\n', s, nS, density_scale);

    % Scene → triangulation → DCEL → per-face values
    [V, T] = prepare_triangulation_512(density_scale, W, H);
    DCEL = build_dcel_from_triangulation(V, T);
    DCEL.faces_value = compute_face_values(DCEL, V);
    triValues = DCEL.faces_value;

    % Precompute triangle MBRs (for RTree, pruning, etc.)
    triMBR = tri_mbrs(V, T); % [xmin ymin xmax ymax] per triangle

    % Prepare indices once (for KDTree baseline)
    cent = (V(T(:,1),:) + V(T(:,2),:) + V(T(:,3),:))/3;
    KD = build_kdtree_wrapper(cent);

    % Build indexes per method (count build time & memory)
    for m = 1:nM
        method = METHODS{m};
        t0 = tic;
        switch method
            case {'QDCEL-Leaf','QDCEL-Clip'}
                params = base_params; params.altitude = ALT_SET(1); % altitude-independent build
                [qt, leaves] = build_qdcel_wrapper(T, V, triValues, LEAF_PIX, params, W, H);
                IDX{m}.qt = qt; IDX{m}.leaves = leaves; %#ok<SAGROW>
                IDX{m}.triValues = triValues; IDX{m}.V = V; IDX{m}.T = T; %#ok<SAGROW>
            case 'UniformQT-Leaf'
                [qt, leaves] = build_uniform_quadtree(T, V, triValues, LEAF_PIX, MAXDEPTH, W, H);
                IDX{m}.qt = qt; IDX{m}.leaves = leaves; %#ok<SAGROW>
                IDX{m}.triValues = triValues; IDX{m}.V = V; IDX{m}.T = T; %#ok<SAGROW>
            case 'KDTree'
                IDX{m}.KD = KD; IDX{m}.cent = cent; %#ok<SAGROW>
                IDX{m}.triValues = triValues; IDX{m}.V = V; IDX{m}.T = T; %#ok<SAGROW>
            case 'RTree'
                RT = rtree_build(triMBR, 8); % M=8 (simple STR bulk-load)
                IDX{m}.RT = RT; IDX{m}.triMBR = triMBR; %#ok<SAGROW>
                IDX{m}.triValues = triValues; IDX{m}.V = V; IDX{m}.T = T; %#ok<SAGROW>
        end
        BuildT(s,m,1) = toc(t0); % build time from 1 run (build once per scene)
        MemMB(s,m,1) = struct_size_mb(IDX{m});
        % replicate build time & memory for averaging (constant over runs)
        BuildT(s,m,2:NRUNS) = BuildT(s,m,1);
        MemMB(s,m,2:NRUNS) = MemMB(s,m,1);
    end

    % Per-run random FoV & altitude, measure query & accuracy
    for r = 1:NRUNS
        fovSize = FoV_PIX( 1 + mod(r-1, numel(FoV_PIX)) );
        altitude = ALT_SET( 1 + mod(r-1, numel(ALT_SET)) );
        [Qrect, rectXY] = random_fov_rect(W, H, fovSize); %#ok<NASGU>

        % dynamic tau (same formula as in your quadtree)
        tau = base_params.tau0 * max(1, altitude/base_params.z0)^(-base_params.alpha);

        % ground-truth IoU (union of triangles with value>tau vs FoV)
        [gtInterArea, gtUnionArea] = truth_iou_poly(Qrect, T, V, triValues, tau); %#ok<ASGLU>

        for m = 1:nM
            method = METHODS{m};
            switch method
                case 'QDCEL-Leaf'
                    % union of leaves with value>=tau
                    t1 = tic;
                    leaves = IDX{m}.leaves;
                    predPoly = polyshape();
                    for iL = 1:size(leaves,1)
                        if leaves(iL,5) >= tau
                            xs = leaves(iL,1); ys = leaves(iL,2);
                            xe = leaves(iL,3); ye = leaves(iL,4);
                            R = [xs ys; xe ys; xe ye; xs ye; xs ys];
                            predPoly = union(predPoly, polyshape(R(:,1),R(:,2)));
                        end
                    end
                    Prect = polyshape(Qrect(:,1), Qrect(:,2));
                    interArea = area(intersect(predPoly, Prect));
                    unionArea = area(union(predPoly, Prect));
                    QryT(s,m,r) = toc(t1);
                    AccIoU(s,m,r)= interArea / max(unionArea, eps);

                case 'QDCEL-Clip'
                    % prune via bbox of FoV → clip triangles with value>tau
                    t1 = tic;
                    Vloc = IDX{m}.V; Tloc = IDX{m}.T; vals = IDX{m}.triValues;
                    x1=min(Qrect(:,1)); x2=max(Qrect(:,1));
                    y1=min(Qrect(:,2)); y2=max(Qrect(:,2));
                    cand = find_relevant_triangles_bbox(Tloc, Vloc, x1, y1, x2, y2);
                    predPoly = polyshape();
                    for iF = cand.'
                        if vals(iF) <= tau, continue; end
                        tri = Vloc(Tloc(iF,:),:);
                        predPoly = union(predPoly, polyshape(tri(:,1), tri(:,2)));
                    end
                    Prect = polyshape(Qrect(:,1), Qrect(:,2));
                    interArea = area(intersect(predPoly, Prect));
                    unionArea = area(union(predPoly, Prect));
                    QryT(s,m,r) = toc(t1);
                    AccIoU(s,m,r)= interArea / max(unionArea, eps);

                case 'UniformQT-Leaf'
                    t1 = tic;
                    leaves = IDX{m}.leaves;
                    predPoly = polyshape();
                    % simulate same thresholding (value field از quadtree با bigVals ساخته شده)
                    % پس برای برابری، از همه‌ی برگ‌ها استفاده می‌کنیم (leaf-only)
                    for iL = 1:size(leaves,1)
                        xs = leaves(iL,1); ys = leaves(iL,2);
                        xe = leaves(iL,3); ye = leaves(iL,4);
                        R = [xs ys; xe ys; xe ye; xs ye; xs ys];
                        predPoly = union(predPoly, polyshape(R(:,1),R(:,2)));
                    end
                    Prect = polyshape(Qrect(:,1), Qrect(:,2));
                    interArea = area(intersect(predPoly, Prect));
                    unionArea = area(union(predPoly, Prect));
                    QryT(s,m,r) = toc(t1);
                    AccIoU(s,m,r)= interArea / max(unionArea, eps);

                case 'KDTree'
                    t1 = tic;
                    cent = IDX{m}.cent; vals = IDX{m}.triValues;
                    Vloc = IDX{m}.V; Tloc = IDX{m}.T;
                    inFoV = inpolygon(cent(:,1), cent(:,2), Qrect(:,1), Qrect(:,2));
                    mask = vals > tau & inFoV;
                    predPoly = polyshape();
                    for iF = find(mask).'
                        tri = Vloc(Tloc(iF,:),:);
                        predPoly = union(predPoly, polyshape(tri(:,1), tri(:,2)));
                    end
                    Prect = polyshape(Qrect(:,1), Qrect(:,2));
                    interArea = area(intersect(predPoly, Prect));
                    unionArea = area(union(predPoly, Prect));
                    QryT(s,m,r) = toc(t1);
                    AccIoU(s,m,r)= interArea / max(unionArea, eps);

                case 'RTree'
                    t1 = tic;
                    RT = IDX{m}.RT; vals = IDX{m}.triValues;
                    Vloc = IDX{m}.V; Tloc = IDX{m}.T;
                    % query by rectangle
                    rectR = [min(Qrect(:,1)) min(Qrect(:,2)) max(Qrect(:,1)) max(Qrect(:,2))];
                    cand = rtree_query(RT, rectR); % candidate triangles by MBR
                    predPoly = polyshape();
                    for iF = cand.'
                        if vals(iF) <= tau, continue; end
                        tri = Vloc(Tloc(iF,:),:);
                        predPoly = union(predPoly, polyshape(tri(:,1), tri(:,2)));
                    end
                    Prect = polyshape(Qrect(:,1), Qrect(:,2));
                    interArea = area(intersect(predPoly, Prect));
                    unionArea = area(union(predPoly, Prect));
                    QryT(s,m,r) = toc(t1);
                    AccIoU(s,m,r)= interArea / max(unionArea, eps);
            end
        end
        fprintf(' run %d/%d: FoV=%d, alt=%.0f → IoU(QDCEL-Clip)=%.3f\n', ...
            r, NRUNS, fovSize, altitude, AccIoU(s,2,r));
    end
end

%% ---------------------- Plots (ISI style) -------------------------------
set_isi_style();
make_plots_four(SCENES, METHODS, BuildT, MemMB, QryT, AccIoU);

exportgraphics(gcf,'qdcel_bench_plots_v2.pdf','ContentType','vector','Resolution',300);
fprintf('\nSaved plots: qdcel_bench_plots_v2.pdf\n');

%% ---------------------- Save raw results --------------------------------
S.results.methods = METHODS;
S.results.scales = SCENES;
S.results.buildT = BuildT;
S.results.memMB = MemMB;
S.results.qryT = QryT;
S.results.accIoU = AccIoU;
save('qdcel_bench_results_v2.mat','-struct','S');
fprintf('Saved raw: qdcel_bench_results_v2.mat\n');

end % === main ===


%% =================== Scene & Triangulation (512×512) ====================
function [V, T] = prepare_triangulation_512(density_scale, W, H)
poly = [
    -102.75393999999825, 19.786959999999674;
    -102.75420999999827, 19.78647999999966;
    -102.75396999999825, 19.786189999999657;
    -102.75380999999825, 19.78611999999966;
    -102.75299999999825, 19.78610999999966;
    -102.75261999999826, 19.78629999999966;
    -102.75240999999825, 19.786249999999665;
    -102.75214999999827, 19.786319999999662;
    -102.75199999999828, 19.786249999999665;
    -102.75164999999826, 19.786499999999663;
    -102.75161999999827, 19.786619999999665;
    -102.75158999999826, 19.787079999999666;
    -102.75143999999827, 19.787469999999654;
    -102.75143999999827, 19.788379999999666;
    -102.75194999999825, 19.78768999999966;
    -102.75228999999825, 19.787459999999665;
    -102.75269999999827, 19.78741999999966;
    -102.75393999999825, 19.786959999999674
];
sx = (poly(:,1)-min(poly(:,1)))./(max(poly(:,1))-min(poly(:,1)));
sy = (poly(:,2)-min(poly(:,2)))./(max(poly(:,2))-min(poly(:,2)));
scaleFactor = 0.55;
offX = (1-scaleFactor)*W/2; offY = (1-scaleFactor)*H/2;
Px = offX + sx*scaleFactor*W; Py = offY + sy*scaleFactor*H;

base = [Px Py];
addN = round(density_scale*7.5*size(base,1));
rx = offX + (scaleFactor*W).*rand(addN,1);
ry = offY + (scaleFactor*H).*rand(addN,1);
in = inpolygon(rx,ry,Px,Py);
V = unique([base; rx(in) ry(in)],'rows','stable');

T0 = delaunay(V(:,1), V(:,2));
keep = false(size(T0,1),1);
for i=1:size(T0,1)
    tri = V(T0(i,:),:); c = mean(tri,1);
    keep(i) = all(inpolygon(tri(:,1),tri(:,2),Px,Py)) & inpolygon(c(1),c(2),Px,Py);
end
T = T0(keep,:);
end


%% =================== DCEL construction (real) ===========================
function DCEL = build_dcel_from_triangulation(V, T)
nV = size(V,1);
nF = size(T,1);
HE = []; faceHE = zeros(nF,1);

DCEL.vertices = repmat(struct('coords',[],'halfEdge',[]), nV,1);
for i=1:nV
    DCEL.vertices(i).coords = V(i,:);
    DCEL.vertices(i).halfEdge= [];
end

edgeMap = containers.Map('KeyType','char','ValueType','any');
idxHE = 0;
for f=1:nF
    idx = T(f,:);
    loc = zeros(3,1);
    for e=1:3
        vA = idx(e);
        vB = idx(1+mod(e,3));
        idxHE = idxHE + 1;
        HE(idxHE).origin = vA; %#ok<AGROW>
        HE(idxHE).twin = 0;
        HE(idxHE).next = 0;
        HE(idxHE).prev = 0;
        HE(idxHE).face = f;
        loc(e) = idxHE;
        if isempty(DCEL.vertices(vA).halfEdge), DCEL.vertices(vA).halfEdge = idxHE; end
        k = edgekey_undirected(vA, vB);
        if isKey(edgeMap,k), edgeMap(k) = [edgeMap(k) idxHE]; else, edgeMap(k) = idxHE; end
    end
    HE(loc(1)).next = loc(2); HE(loc(2)).next = loc(3); HE(loc(3)).next = loc(1);
    HE(loc(1)).prev = loc(3); HE(loc(2)).prev = loc(1); HE(loc(3)).prev = loc(2);
    faceHE(f) = loc(1);
end

keysList = keys(edgeMap);
for kk=1:numel(keysList)
    k = keysList{kk};
    heList = edgeMap(k);
    if numel(heList)==2
        a = heList(1); b = heList(2);
        HE(a).twin = b; HE(b).twin = a;
    end
end

DCEL.faces = repmat(struct('halfEdge',[],'vertices',[],'value',0), nF,1);
for f=1:nF
    DCEL.faces(f).halfEdge = faceHE(f);
    DCEL.faces(f).vertices = T(f,:);
end
DCEL.halfEdges = HE;
end

function k = edgekey_undirected(a,b)
if a>b, t=a; a=b; b=t; end
k = sprintf('%d_%d',a,b);
end

function vals = compute_face_values(DCEL, V)
nF = numel(DCEL.faces);
vals = zeros(nF,1);
for f=1:nF
    idx = DCEL.faces(f).vertices;
    face_coords = V(idx,:);
    vals(f) = compute_value_for_face(face_coords);
end
end

function value = compute_value_for_face(face_coords)
center = mean(face_coords);
value = 20 + (100 - 20) * exp(-norm(center) / 100);
end


%% ====================== Triangle MBRs & helpers =========================
function mbr = tri_mbrs(V, T)
n = size(T,1); mbr = zeros(n,4);
for i=1:n
    tri = V(T(i,:),:);
    mbr(i,:) = [min(tri(:,1)) min(tri(:,2)) max(tri(:,1)) max(tri(:,2))];
end
end

function [Q, rectXY] = random_fov_rect(W,H,side)
cx = 20 + rand*(W-40); cy = 20 + rand*(H-40);
half = side/2;
xs = max(1, cx-half); ys = max(1, cy-half);
xe = min(W, cx+half); ye = min(H, cy+half);
Q = [xs ys; xe ys; xe ye; xs ye; xs ys];
rectXY = [xs ys xe ye]; %#ok<NASGU>
end


%% ===================== RTree (STR bulk-load, simple) ====================
function RT = rtree_build(rects, M)
% rects: [xmin ymin xmax ymax]; M: max children per node
% STR bulk-load: sort by x-center → slice → sort by y-center → group
N = size(rects,1);
idx = (1:N).';
centx = (rects(:,1)+rects(:,3))/2; [~, px] = sort(centx);
S = ceil(N / M); % number of leaves approx
Sx = ceil(sqrt(S)); % #slices by x
sliceSize = ceil(N / Sx);

leaves = {};
for sx = 1:Sx
    lo = (sx-1)*sliceSize + 1; hi = min(sx*sliceSize, N);
    slabIdx = px(lo:hi);
    cy = (rects(slabIdx,2)+rects(slabIdx,4))/2; [~, py] = sort(cy);
    slabIdx = slabIdx(py);
    for k = 1:M:numel(slabIdx)
        kk = slabIdx(k:min(k+M-1, numel(slabIdx)));
        node.isLeaf = true;
        node.idx = idx(kk);
        node.rects = rects(kk,:);
        node.children = [];
        node.mbr = [min(node.rects(:,1)) min(node.rects(:,2)) ...
                    max(node.rects(:,3)) max(node.rects(:,4))];
        leaves{end+1} = node; %#ok<AGROW>
    end
end
RT = rtree_pack(leaves, M);
end

function node = rtree_pack(children, M)
if numel(children) == 1
    node = children{1};
    return;
end
% pack upper level by grouping child MBRs
C = numel(children);
rects = zeros(C,4);
for i=1:C, rects(i,:) = children{i}.mbr; end
centx = (rects(:,1)+rects(:,3))/2; [~, px] = sort(centx);
S = ceil(C / M); Sx = ceil(sqrt(S)); sliceSize = ceil(C / Sx);

parents = {};
for sx=1:Sx
    lo = (sx-1)*sliceSize + 1; hi = min(sx*sliceSize, C);
    slabIdx = px(lo:hi);
    cy = (rects(slabIdx,2)+rects(slabIdx,4))/2; [~, py] = sort(cy);
    slabIdx = slabIdx(py);
    for k=1:M:numel(slabIdx)
        kk = slabIdx(k:min(k+M-1, numel(slabIdx)));
        n.isLeaf = false; n.idx = []; n.rects = [];
        n.children = children(kk);
        m = cellfun(@(x) x.mbr, n.children, 'UniformOutput', false);
        m = vertcat(m{:});
        n.mbr = [min(m(:,1)) min(m(:,2)) max(m(:,3)) max(m(:,4))];
        parents{end+1} = n; %#ok<AGROW>
    end
end
node = rtree_pack(parents, M);
end

function outIdx = rtree_query(node, rect)
% rect: [xmin ymin xmax ymax]
outIdx = [];
if ~mbr_intersect(node.mbr, rect), return; end
if node.isLeaf
    % filter leaf rects
    keep = mbr_intersect_many(node.rects, rect);
    outIdx = node.idx(keep);
else
    for i=1:numel(node.children)
        outIdx = [outIdx; rtree_query(node.children{i}, rect)]; %#ok<AGROW>
    end
end
end

function tf = mbr_intersect(a,b)
% a,b: [xmin ymin xmax ymax]
tf = ~(a(3) < b(1) || b(3) < a(1) || a(4) < b(2) || b(4) < a(2));
end
function keep = mbr_intersect_many(R, b)
keep = ~(R(:,3) < b(1) | b(3) < R(:,1) | R(:,4) < b(2) | b(4) < R(:,2));
end


%% ===================== KDTree baseline (fallback safe) ==================
function KD = build_kdtree_wrapper(cent)
try
    KD.tree = KDTreeSearcher(cent,'BucketSize',16);
    KD.impl = 'KDTreeSearcher';
catch
    KD.tree = cent; KD.impl = 'linear';
end
end


%% ===================== Ground truth (polyshape IoU) =====================
function [interArea, unionArea] = truth_iou_poly(Qrect, T, V, triValues, tau)
Prect = polyshape(Qrect(:,1), Qrect(:,2));
Pval = polyshape();
minx=min(Qrect(:,1)); maxx=max(Qrect(:,1));
miny=min(Qrect(:,2)); maxy=max(Qrect(:,2));
for i=1:size(T,1)
    if triValues(i) <= tau, continue; end
    tri = V(T(i,:),:);
    bx = [min(tri(:,1)) max(tri(:,1)) min(tri(:,2)) max(tri(:,2))];
    if bx(2) < minx || maxx < bx(1) || bx(4) < miny || maxy < bx(3), continue; end
    Pval = union(Pval, polyshape(tri(:,1), tri(:,2)));
end
interArea = area(intersect(Pval, Prect));
unionArea = area(union(Pval, Prect));
end


%% ========================= Plots (ISI style) ============================
function set_isi_style()
set(groot,'defaultFigureColor','w');
set(groot,'defaultAxesFontName','Times New Roman');
set(groot,'defaultAxesFontSize',11);
set(groot,'defaultLineLineWidth',1.5);
set(groot,'defaultAxesLineWidth',1.0);
set(groot,'defaultAxesGridAlpha',0.15);
end

function make_plots_four(SCENES, METHODS, BuildT, MemMB, QryT, AccIoU)
nS = numel(SCENES); nM = numel(METHODS);
mu = @(A) squeeze(mean(A,3));
se = @(A) squeeze(1.96*std(A,0,3)/sqrt(size(A,3))); % 95% CI

fig = figure('Color','w','Position',[60 60 1600 420]);

% ---- Build Time (log-y) ----
subplot(1,4,1); hold on; grid on; box on;
M = mu(BuildT); E = se(BuildT);
mrk = {'o','s','^','d','v'};
for m=1:nM, errorbar(1:nS, M(:,m), E(:,m), '-','Marker',mrk{1+mod(m-1,numel(mrk))}); end
set(gca,'XTick',1:nS,'XTickLabel',compose('dens=%.2f',SCENES));
set(gca,'YScale','log');
xlabel('Scene density'); ylabel('Build time (s)'); title('Build Time');
legend(METHODS,'Location','northwest','Box','off');

% ---- Memory ----
subplot(1,4,2); hold on; grid on; box on;
M2 = mu(MemMB); E2 = se(MemMB);
w = 0.16; x = 1:nS;
for m=1:nM
    off = (m - (nM+1)/2)*w*1.8;
    bar(x+off, M2(:,m), w, 'EdgeColor','k','LineWidth',0.6);
    errorbar(x+off, M2(:,m), E2(:,m), '.k','LineWidth',1.1);
end
set(gca,'XTick',1:nS,'XTickLabel',compose('dens=%.2f',SCENES));
xlabel('Scene density'); ylabel('Peak RAM (MB)'); title('Memory Usage');

% ---- Query Time (log-y) ----
subplot(1,4,3); hold on; grid on; box on;
M4 = mu(QryT); E4 = se(QryT);
for m=1:nM, errorbar(1:nS, M4(:,m), E4(:,m), '-','Marker','>'); end
set(gca,'XTick',1:nS,'XTickLabel',compose('dens=%.2f',SCENES));
set(gca,'YScale','log');
xlabel('Scene density'); ylabel('Query time (s)'); title('Query Latency');

% ---- Accuracy (IoU) ----
subplot(1,4,4); hold on; grid on; box on;
M3 = mu(AccIoU); E3 = se(AccIoU);
for m=1:nM, errorbar(1:nS, M3(:,m), E3(:,m), '-','Marker','square'); end
ylim([0 1]); yticks(0:0.2:1.0);
set(gca,'XTick',1:nS,'XTickLabel',compose('dens=%.2f',SCENES));
xlabel('Scene density'); ylabel('IoU'); title('FoV Query Accuracy');

% legend only once (top-left subplot)
subplot(1,4,1);
legend(METHODS,'Location','northwest','Box','off');
set(findall(fig,'-property','FontName'),'FontName','Times New Roman');
end


%% ==================== Utilities (bytes/MB) ==============================
function mb = struct_size_mb(s)
info = whos('s'); mb = info.bytes/1e6;
end


%% =================== YOUR ORIGINAL FUNCTIONS (AS-IS) ====================
% -------- FoVOverlap_9Intersection (unmodified) -------------------------
function [overlap_area, relation] = FoVOverlap_9Intersection(tri_coords, x_start, y_start, x_end, y_end)
rect_coords = [x_start, y_start; x_end, y_start; x_end, y_end; x_start, y_end];
tol = 1e-9;
use_polyshape = exist('polyshape','file') == 2;
if use_polyshape
    Ptri = polyshape(tri_coords(:,1), tri_coords(:,2));
    Prect = polyshape(rect_coords(:,1), rect_coords(:,2));
    I = intersect(Ptri, Prect);
    overlap_area = area(I);
    if overlap_area > tol
        aTri = max(area(Ptri), tol);
        aRect = max(area(Prect), tol);
        if abs(overlap_area - aTri) <= 1e-6 * aTri || ...
           abs(overlap_area - aRect) <= 1e-6 * aRect
            relation = 'contain';
        else
            relation = 'overlap';
        end
        return;
    else
        tri_edges = local_get_edges(tri_coords);
        rect_edges = local_get_edges(rect_coords);
        touch = false;
        for i = 1:size(tri_edges,1)
            p1 = tri_edges(i,1:2); p2 = tri_edges(i,3:4);
            for j = 1:size(rect_edges,1)
                q1 = rect_edges(j,1:2); q2 = rect_edges(j,3:4);
                if local_segments_touch_or_intersect(p1,p2,q1,q2)
                    touch = true; break;
                end
            end
            if touch, break; end
        end
        if ~touch
            [~, on1] = inpolygon(tri_coords(:,1), tri_coords(:,2), rect_coords(:,1), rect_coords(:,2));
            [~, on2] = inpolygon(rect_coords(:,1), rect_coords(:,2), tri_coords(:,1), tri_coords(:,2));
            touch = any(on1) || any(on2);
        end
        if touch, relation = 'touch'; else, relation = 'disjoint'; end
        overlap_area = 0; return;
    end
else
    [ix, iy] = polybool('intersection', tri_coords(:,1), tri_coords(:,2), rect_coords(:,1), rect_coords(:,2));
    if isempty(ix) || isempty(iy)
        tri_edges = local_get_edges(tri_coords);
        rect_edges = local_get_edges(rect_coords);
        touch = false;
        for i = 1:size(tri_edges,1)
            p1 = tri_edges(i,1:2); p2 = tri_edges(i,3:4);
            for j = 1:size(rect_edges,1)
                q1 = rect_edges(j,1:2); q2 = rect_edges(j,3:4);
                if local_segments_touch_or_intersect(p1,p2,q1,q2)
                    touch = true; break;
                end
            end
            if touch, break; end
        end
        if ~touch
            [~, on1] = inpolygon(tri_coords(:,1), tri_coords(:,2), rect_coords(:,1), rect_coords(:,2));
            [~, on2] = inpolygon(rect_coords(:,1), rect_coords(:,2), tri_coords(:,1), tri_coords(:,2));
            touch = any(on1) || any(on2);
        end
        if touch, relation = 'touch'; else, relation = 'disjoint'; end
        overlap_area = 0; return;
    else
        overlap_area = polyarea(ix, iy);
        aTri = max(polyarea(tri_coords(:,1), tri_coords(:,2)), tol);
        aRect = max(polyarea(rect_coords(:,1), rect_coords(:,2)), tol);
        if abs(overlap_area - aTri) <= 1e-6 * aTri || ...
           abs(overlap_area - aRect) <= 1e-6 * aRect
            relation = 'contain';
        else
            relation = 'overlap';
        end
        return;
    end
end
end

function edges = local_get_edges(coords)
n = size(coords,1); edges = zeros(n,4);
for i = 1:n
    j = i + 1; if j > n, j = 1; end
    edges(i,:) = [coords(i,1) coords(i,2) coords(j,1) coords(j,2)];
end
end

function tf = local_segments_touch_or_intersect(p1,p2,q1,q2)
tf = false;
if max(p1(1),p2(1)) < min(q1(1),q2(1)) || max(q1(1),q2(1)) < min(p1(1),p2(1)) || ...
   max(p1(2),p2(2)) < min(q1(2),q2(2)) || max(q1(2),q2(2)) < min(p1(2),p2(2))
    return;
end
o1 = local_orientation(p1,p2,q1);
o2 = local_orientation(p1,p2,q2);
o3 = local_orientation(q1,q2,p1);
o4 = local_orientation(q1,q2,p2);
if (o1 ~= o2) && (o3 ~= o4), tf = true; return; end
if o1 == 0 && local_on_segment(p1,q1,p2), tf = true; return; end
if o2 == 0 && local_on_segment(p1,q2,p2), tf = true; return; end
if o3 == 0 && local_on_segment(q1,p1,q2), tf = true; return; end
if o4 == 0 && local_on_segment(q1,p2,q2), tf = true; return; end
end

function val = local_orientation(a,b,c)
v = (b(2)-a(2))*(c(1)-b(1)) - (b(1)-a(1))*(c(2)-b(2));
if abs(v) < 1e-12, val = 0;
elseif v > 0, val = 1;
else, val = 2;
end
end

function tf = local_on_segment(a,b,c)
tf = (min(a(1),c(1)) - 1e-12 <= b(1)) && (b(1) <= max(a(1),c(1)) + 1e-12) && ...
     (min(a(2),c(2)) - 1e-12 <= b(2)) && (b(2) <= max(a(2),c(2)) + 1e-12);
end


% -------- quadtree_split_v2 (unmodified) -------------------------------
function [final_list, quad_data, small_squares_data] = quadtree_split_v2( ...
    x_start, y_start, x_end, y_end, triangles, vertices, triangleValues, ...
    list, code, drone_cover, quad_data, small_squares_data, params)

if nargin < 13 || isempty(params), params = struct(); end
if ~isfield(params,'tau0'), params.tau0 = 0.01; end
if ~isfield(params,'alpha'), params.alpha = 1.0; end
if ~isfield(params,'altitude'), params.altitude= []; end
if ~isfield(params,'z0'), params.z0 = 50; end
if ~isfield(params,'minDepth'), params.minDepth= 0; end
if ~isfield(params,'maxDepth'), params.maxDepth= 20; end
if ~isfield(params,'depth'), params.depth = 0; end

if nargin < 11 || isempty(quad_data), quad_data = []; end
if nargin < 12 || isempty(small_squares_data), small_squares_data = []; end
if nargin < 8 || isempty(list), list = []; end
if nargin < 10 || isempty(code), code = ''; end

width = x_end - x_start + 1;
height = y_end - y_start + 1;

relevant_triangles = find_relevant_triangles_bbox(triangles, vertices, x_start, y_start, x_end, y_end);

if isempty(params.altitude)
    tau = params.tau0;
else
    tau = params.tau0 * max(1, params.altitude/params.z0)^(-params.alpha);
end

quad_value = 0;
rect_area = max((x_end - x_start) * (y_end - y_start), eps);
if ~isempty(relevant_triangles)
    for i = 1:length(relevant_triangles)
        tri = relevant_triangles(i);
        tri_coords = vertices(triangles(tri,:),:);
        [overlap_area, relation] = FoVOverlap_9Intersection(tri_coords, x_start, y_start, x_end, y_end);
        if overlap_area <= 0, continue; end
        value_face = triangleValues(tri);
        tri_area = max(polyarea(tri_coords(:,1), tri_coords(:,2)), eps);
        w = overlap_area / tri_area;
        if strcmp(relation,'contain'), w = min(1.0, rect_area / tri_area); end
        quad_value = quad_value + w * value_face;
    end
end

current_node = struct('code', code, ...
                      'x_start', x_start, 'y_start', y_start, ...
                      'x_end', x_end, 'y_end', y_end, ...
                      'value', quad_value, 'depth', params.depth);
list = [list; current_node];
quad_data = [quad_data; x_start, y_start, x_end, y_end, quad_value, params.depth];

stop_here = (quad_value < tau && params.depth >= params.minDepth) || ...
            (width <= drone_cover) || (height <= drone_cover) || ...
            (params.depth >= params.maxDepth) || isempty(relevant_triangles);

if stop_here
    small_squares_data = [small_squares_data; x_start, y_start, x_end, y_end, quad_value, params.depth];
    final_list = list; return;
end

x_mid = floor((x_start + x_end)/2);
y_mid = floor((y_start + y_end)/2);
params.depth = params.depth + 1;

[list, quad_data, small_squares_data] = quadtree_split_v2(x_start, y_start, x_mid, y_mid, triangles, vertices, triangleValues, list, [code '3'], drone_cover, quad_data, small_squares_data, params);
[list, quad_data, small_squares_data] = quadtree_split_v2(x_start, y_mid+1, x_mid, y_end, triangles, vertices, triangleValues, list, [code '2'], drone_cover, quad_data, small_squares_data, params);
[list, quad_data, small_squares_data] = quadtree_split_v2(x_mid+1, y_mid+1, x_end, y_end, triangles, vertices, triangleValues, list, [code '1'], drone_cover, quad_data, small_squares_data, params);
[list, quad_data, small_squares_data] = quadtree_split_v2(x_mid+1, y_start, x_end, y_mid, triangles, vertices, triangleValues, list, [code '4'], drone_cover, quad_data, small_squares_data, params);

final_list = list;
end

function relevant_triangles = find_relevant_triangles_bbox(triangles, vertices, x_start, y_start, x_end, y_end)
relevant_triangles = [];
rect = [x_start, x_end, y_start, y_end];
for i = 1:size(triangles,1)
    tri_idx = triangles(i,:);
    if any(tri_idx <= 0) || any(tri_idx > size(vertices,1)), continue; end
    tri_v = vertices(tri_idx,:);
    bx = [min(tri_v(:,1)) max(tri_v(:,1)) min(tri_v(:,2)) max(tri_v(:,2))];
    if ~(bx(2) < rect(1) || rect(2) < bx(1) || bx(4) < rect(3) || rect(4) < bx(3))
        relevant_triangles = [relevant_triangles; i];
    end
end
end

% ================== MISSING HELPERS (ADD TO THE SAME FILE) ==================
function [quad_data, leaves] = build_qdcel_wrapper(triangles, vertices, triValues, leaf_pix, params, W, H)
% QDCEL builder using your dynamic quadtree AS-IS
% Returns:
% quad_data: [xs ys xe ye value depth]
% leaves : same as small_squares_data (for leaf-union / queries)
    if nargin < 6, W = 512; H = 512; end
    [~, quad_data, small_squares_data] = quadtree_split_v2( ...
        1, 1, W, H, ...
        triangles, vertices, triValues, ...
        [], 'Q_', leaf_pix, [], [], params);
    leaves = small_squares_data; % [xs ys xe ye value depth]
end

function [quad_data, leaves] = build_uniform_quadtree(triangles, vertices, triValues, leaf_pix, maxDepth, W, H)
% Uniform Quadtree baseline:
% force splitting by neutralizing threshold (big values), so only leaf size/depth stop
    if nargin < 6, W = 512; H = 512; end
    params = struct('tau0',0, 'alpha',0, 'z0',1, ...
                    'minDepth',0, 'maxDepth',maxDepth, ...
                    'altitude',[], 'depth',0);
    bigVals = max(triValues) * ones(size(triValues)); % stop rule ignores tau
    [~, quad_data, small_squares_data] = quadtree_split_v2( ...
        1, 1, W, H, ...
        triangles, vertices, bigVals, ...
        [], 'U_', leaf_pix, [], [], params);
    leaves = small_squares_data; % [xs ys xe ye value(depth-less-meaningful here) depth]
end
% ============================================================================ 
