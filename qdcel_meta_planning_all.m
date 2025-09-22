function qdcel_meta_planning_all()

clc; clear; close all; rng(42);

%% ------------------ Config ------------------
W=512; H=512; dens=1.0;                    % scene
FoV_side = 96;                              % FoV size (px)
ALT = 60;                                   % altitude (m)
leaf_px = 32;                               % quadtree leaf
params = struct('tau0',0.10,'alpha',1.25,'z0',50,'minDepth',0,'maxDepth',12,'altitude',ALT);

% metaheuristic config
nVisit = 25;        % number of candidate points to visit between start/goal
lambda = 0.75;      % trade-off: length - lambda * sum(value)
pop    = 24;        % population size
maxIt  = 80;        % iterations

%% ------------- Scene, DCEL, values -------------
[V,T]  = prepare_triangulation_512(dens, W, H);
DCEL   = build_dcel_from_triangulation(V, T);
triVal = compute_face_values(DCEL, V);       % scalar per-face value (from your model)

% dynamic threshold tau(h)
tau = params.tau0 * max(1, ALT/params.z0)^(-params.alpha);

% FoV rectangle (random but reproducible)
[Qrect, ~] = random_fov_rect(W, H, FoV_side);

% ---------- Build indexes ----------
% QDCEL (quadtree) for pruning only; prediction uses triangle-level clip
[~, ~]  = build_qdcel_wrapper(T, V, triVal, leaf_px, params, W, H);  % (kept if later needed)
% KDTree (centroids)
cent = (V(T(:,1),:)+V(T(:,2),:)+V(T(:,3),:))/3;
KD   = build_kdtree_wrapper(cent);
% RTree (triangle MBRs)
triMBR = tri_mbrs(V,T);
RT     = rtree_build(triMBR, 8);

%% ---------- Candidate extraction by index ----------
IDXNAMES = {'QDCEL-Clip','KDTree','RTree'};
nI = numel(IDXNAMES);

C.allResults = struct([]);

for iI = 1:nI
    name = IDXNAMES{iI};
    switch name
        case 'QDCEL-Clip'
            [candIdx] = select_candidates_qdcel_clip(Qrect, T, V, triVal, tau);
        case 'KDTree'
            [candIdx] = select_candidates_kdtree(Qrect, T, V, triVal, tau, KD);
        case 'RTree'
            [candIdx] = select_candidates_rtree(Qrect, T, V, triVal, tau, RT, triMBR);
    end

    % Fallback: if few candidates, relax filters
    if numel(candIdx) < 8
        % take top by value in FoV bbox regardless of inside tests
        bbox = [min(Qrect(:,1)) min(Qrect(:,2)) max(Qrect(:,1)) max(Qrect(:,2))];
        allIdx = find_by_bbox(T,V,bbox);
        [~,ord] = sort(triVal(allIdx),'descend');
        candIdx = unique([candIdx(:); allIdx(ord(1:min(200,numel(ord))))(:)]);
        candIdx = candIdx(triVal(candIdx) > prctile(triVal(candIdx), 50));
    end

    candPts = (V(T(candIdx,1),:)+V(T(candIdx,2),:)+V(T(candIdx,3),:))/3;
    candVal = triVal(candIdx);

    % Normalize values to [0,1] for fitness
    if isempty(candVal)
        warning('%s: no candidates; skipping', name);
        continue;
    end
    candValN = normalize01(candVal);

    % pick start/goal as farthest pair among top-k by value (robust)
    kTop = min(50, size(candPts,1));
    [~,topOrd] = sort(candVal,'descend');
    Pk = candPts(topOrd(1:kTop),:);
    [sXY,gXY] = farthest_pair(Pk);

    % ensure nVisit feasible
    nVisitEff = min(nVisit, size(candPts,1));
    if nVisitEff < 3
        nVisitEff = min(3, size(candPts,1));
    end

    % ---------- Run metaheuristics (rank-coded, safe) ----------
    [pso_path, pso_hist, pso_fit, pso_cum] = mh_pso_ranked(candPts,candValN,sXY,gXY,nVisitEff,lambda,pop,maxIt);
    [gwo_path, gwo_hist, gwo_fit, gwo_cum] = mh_gwo_ranked(candPts,candValN,sXY,gXY,nVisitEff,lambda,pop,maxIt);
    [ga_path,  ga_hist,  ga_fit,  ga_cum]  = mh_ga_ranked (candPts,candValN,sXY,gXY,nVisitEff,lambda,pop,maxIt);

    % ---------- Plots per index ----------
    fig = figure('Color','w','Position',[60 60 1350 420]);
    set_isi_style();

    % (1) Paths over Triangulation
    subplot(1,3,1); hold on; box on; grid on;
    triplot(delaunay(V(:,1),V(:,2)), V(:,1), V(:,2), 'Color',[0.85 0.85 0.85]);
    scatter(candPts(:,1), candPts(:,2), 12, candValN, 'filled'); colormap(parula); colorbar; caxis([0 1]);
    plot(pso_path(:,1), pso_path(:,2), '-', 'LineWidth',1.6); 
    plot(gwo_path(:,1), gwo_path(:,2), '-', 'LineWidth',1.6); 
    plot(ga_path(:,1),  ga_path(:,2),  '-', 'LineWidth',1.6); 
    plot(sXY(1), sXY(2), 'kp','MarkerFaceColor','k','MarkerSize',9);
    plot(gXY(1), gXY(2), 'ks','MarkerFaceColor','k','MarkerSize',9);
    plot(Qrect(:,1), Qrect(:,2), 'k--','LineWidth',1.0);
    title(sprintf('%s: Metaheuristic Paths on Triangulation', name));
    legend({'Triangulation','Candidates','PSO','GWO','GA','Start','Goal','FoV'},'Location','bestoutside','Box','off');
    axis image tight;

    % (2) Convergence
    subplot(1,3,2); hold on; box on; grid on;
    plot(pso_hist,'-','LineWidth',1.6);
    plot(gwo_hist,'-','LineWidth',1.6);
    plot(ga_hist, '-','LineWidth',1.6);
    xlabel('Iteration'); ylabel('Best fitness (lower is better)');
    title('Convergence'); legend('PSO','GWO','GA','Location','northeast','Box','off');

    % (3) Cumulative value along path
    subplot(1,3,3); hold on; box on; grid on;
    plot(pso_cum,'-','LineWidth',1.6);
    plot(gwo_cum,'-','LineWidth',1.6);
    plot(ga_cum, '-','LineWidth',1.6);
    xlabel('Step'); ylabel('Cumulative normalized value');
    title('Value Accumulation'); legend('PSO','GWO','GA','Location','southeast','Box','off');

    fname = sprintf('paths_convergence_value_%s.pdf', strrep(name,'-','_'));
    exportgraphics(fig, fname, 'ContentType','vector','Resolution',300);
    fprintf('Saved: %s\n', fname);

    % store results
    R.indexName = name;
    R.candCount = size(candPts,1);
    R.startXY = sXY; R.goalXY = gXY;
    R.PSO.path = pso_path; R.PSO.best = pso_fit; R.PSO.conv = pso_hist; R.PSO.cum = pso_cum; R.PSO.totalVal = pso_cum(end);
    R.GWO.path = gwo_path; R.GWO.best = gwo_fit; R.GWO.conv = gwo_hist; R.GWO.cum = gwo_cum; R.GWO.totalVal = gwo_cum(end);
    R.GA.path  = ga_path;  R.GA.best  = ga_fit;  R.GA.conv  = ga_hist;  R.GA.cum  = ga_cum;  R.GA.totalVal  = ga_cum(end);
    C.allResults = [C.allResults; R]; 
end

%% --------- Bar chart: total collected value per algorithm per index ------
if ~isempty(C.allResults)
    algs = {'PSO','GWO','GA'};
    nI = numel(C.allResults);
    M = zeros(nI, numel(algs));
    for i=1:nI
        M(i,1) = C.allResults(i).PSO.totalVal;
        M(i,2) = C.allResults(i).GWO.totalVal;
        M(i,3) = C.allResults(i).GA.totalVal;
    end
    fig2 = figure('Color','w','Position',[80 80 700 360]);
    set_isi_style(); hold on; box on; grid on;
    b = bar(M, 'grouped'); for k=1:numel(b), b(k).EdgeColor='k'; b(k).LineWidth=0.6; end
    set(gca,'XTick',1:nI, 'XTickLabel', {C.allResults.indexName});
    ylabel('Total normalized value collected'); title('Metaheuristics: total value by index');
    legend(algs,'Location','northoutside','Orientation','horizontal','Box','off');
    exportgraphics(fig2, 'total_value_by_index.pdf','ContentType','vector','Resolution',300);
    fprintf('Saved: total_value_by_index.pdf\n');
end

% save raw
save('meta_planning_across_indexes.mat','-struct','C');
fprintf('Saved raw: meta_planning_across_indexes.mat\n');

end 


%% =================== Candidate selection helpers ===================
function idx = select_candidates_qdcel_clip(Qrect, T, V, triVal, tau)
% prune by FoV bbox; then keep triangles with value > tau and intersect FoV
minx=min(Qrect(:,1)); maxx=max(Qrect(:,1));
miny=min(Qrect(:,2)); maxy=max(Qrect(:,2));
cand = find_by_bbox(T,V,[minx miny maxx maxy]);
keep = false(size(cand));
for k=1:numel(cand)
    tri = V(T(cand(k),:),:);
    if triVal(cand(k)) > tau && poly_rect_intersects(tri, [minx miny maxx maxy])
        keep(k)=true;
    end
end
idx = cand(keep);
end

function idx = select_candidates_kdtree(Qrect, T, V, triVal, tau, KD)
cent = (V(T(:,1),:)+V(T(:,2),:)+V(T(:,3),:))/3;
in = inpolygon(cent(:,1), cent(:,2), Qrect(:,1), Qrect(:,2));
idx = find(in & triVal > tau);
% If toolbox KDTreeSearcher available, could use range search; here inpolygon suffices.
end

function idx = select_candidates_rtree(Qrect, T, V, triVal, tau, RT, rects)
rectFoV = [min(Qrect(:,1)) min(Qrect(:,2)) max(Qrect(:,1)) max(Qrect(:,2))];
cand = rtree_query(RT, rectFoV);
cand = unique(cand(:));
if isempty(cand), idx = cand; return; end
keep = false(size(cand));
for k=1:numel(cand)
    if triVal(cand(k)) <= tau, continue; end
    % optional precise test beyond MBR:
    tri = V(T(cand(k),:),:);
    if poly_rect_intersects(tri, rectFoV)
        keep(k)=true;
    end
end
idx = cand(keep);
end


%% =================== Metaheuristics (rank-coded) ===================
% Representation: a real vector w in [0,1]^K. Decode: take top-nVisit indices,
% then order them by nearest-neighbor from start to goal to form a route.

function [bestPathXY, bestHist, bestFit, cumVal] = mh_pso_ranked(P,valN,sXY,gXY,nVisit,lambda,pop,maxIt)
K = size(P,1);
pop = max(6, pop);
maxIt = max(10, maxIt);

X = rand(pop, K);        % particles
V = zeros(pop, K);       % velocities
w_in=0.72; c1=1.49; c2=1.49;

pbest = X; pbestFit = inf(pop,1);
gbest = X(1,:); gbestFit = inf;

bestHist = zeros(maxIt,1);

for it=1:maxIt
    for i=1:pop
        [routeIdx, routeXY] = decode_ranked(X(i,:), P, sXY, gXY, nVisit);
        f = fitness_route(routeXY, routeIdx, valN, lambda);
        if f < pbestFit(i), pbestFit(i) = f; pbest(i,:) = X(i,:); end
        if f < gbestFit,    gbestFit    = f; gbest    = X(i,:);   end
    end
    bestHist(it) = gbestFit;

    % update
    r1 = rand(pop,K); r2 = rand(pop,K);
    V = w_in*V + c1*r1.*(pbest - X) + c2*r2.*(repmat(gbest,pop,1) - X);
    X = X + V;
    X = min(max(X,0),1);
end

[bestRouteIdx, bestPathXY] = decode_ranked(gbest, P, sXY, gXY, nVisit);
bestFit = fitness_route(bestPathXY, bestRouteIdx, valN, lambda);
cumVal  = cumulative_value(bestRouteIdx, valN);
end

function [bestPathXY, bestHist, bestFit, cumVal] = mh_gwo_ranked(P,valN,sXY,gXY,nVisit,lambda,pop,maxIt)
K = size(P,1);
pop = max(6, pop);
X = rand(pop, K);
fit = inf(pop,1);
alpha=X(1,:); beta=X(1,:); delta=X(1,:); fA=inf; fB=inf; fD=inf;

bestHist = zeros(maxIt,1);

for it=1:maxIt
    a = 2 - 2*(it-1)/(maxIt-1);  % linearly decreasing
    % evaluate & update alpha/beta/delta
    for i=1:pop
        [routeIdx, routeXY] = decode_ranked(X(i,:), P, sXY, gXY, nVisit);
        fi = fitness_route(routeXY, routeIdx, valN, lambda); fit(i)=fi;
        if fi < fA
            delta=beta; fD=fB; beta=alpha; fB=fA; alpha=X(i,:); fA=fi;
        elseif fi < fB
            delta=beta; fD=fB; beta=X(i,:); fB=fi;
        elseif fi < fD
            delta=X(i,:); fD=fi;
        end
    end
    bestHist(it) = fA;

    % update positions
    for i=1:pop
        r1 = rand(1,K); r2 = rand(1,K); A1 = 2*a*r1 - a; C1 = 2*r2;
        r1 = rand(1,K); r2 = rand(1,K); A2 = 2*a*r1 - a; C2 = 2*r2;
        r1 = rand(1,K); r2 = rand(1,K); A3 = 2*a*r1 - a; C3 = 2*r2;

        X1 = alpha - A1.*abs(C1.*alpha - X(i,:));
        X2 = beta  - A2.*abs(C2.*beta  - X(i,:));
        X3 = delta - A3.*abs(C3.*delta - X(i,:));

        X(i,:) = (X1 + X2 + X3)/3;
    end
    X = min(max(X,0),1);
end

[bestRouteIdx, bestPathXY] = decode_ranked(alpha, P, sXY, gXY, nVisit);
bestFit = fitness_route(bestPathXY, bestRouteIdx, valN, lambda);
cumVal  = cumulative_value(bestRouteIdx, valN);
end

function [bestPathXY, bestHist, bestFit, cumVal] = mh_ga_ranked(P,valN,sXY,gXY,nVisit,lambda,pop,maxIt)
K = size(P,1);
pop = max(20, pop);           % GA benefits from larger pop
pc = 0.85; pm = 0.15;         % crossover/mutation rates
X = rand(pop, K);
bestHist = zeros(maxIt,1);

for it=1:maxIt
    % evaluate
    f = zeros(pop,1);
    for i=1:pop
        [routeIdx, routeXY] = decode_ranked(X(i,:), P, sXY, gXY, nVisit);
        f(i) = fitness_route(routeXY, routeIdx, valN, lambda);
    end
    [bestHist(it), bidx] = min(f);
    elite = X(bidx,:);

    % selection (tournament)
    newX = zeros(size(X));
    newX(1,:) = elite; % elitism
    for k=2:2:pop
        i1 = tournament(f); i2 = tournament(f);
        j1 = tournament(f); j2 = tournament(f);
        p1 = X(i1,:); p2 = X(i2,:);
        q1 = X(j1,:); q2 = X(j2,:);

        % crossover (blend)
        if rand < pc
            [c1,c2] = blend_crossover(p1,p2);
        else
            c1=p1; c2=p2;
        end
        if rand < pc
            [d1,d2] = blend_crossover(q1,q2);
        else
            d1=q1; d2=q2;
        end

        % mutation (gaussian)
        c1 = mutate_gauss(c1, pm); c2 = mutate_gauss(c2, pm);
        d1 = mutate_gauss(d1, pm); d2 = mutate_gauss(d2, pm);

        newX(k,:) = min(max(c1,0),1);
        if k+1<=pop, newX(k+1,:) = min(max(c2,0),1); end
        if k+2<=pop, newX(k+2,:) = min(max(d1,0),1); end
        if k+3<=pop, newX(k+3,:) = min(max(d2,0),1); end
    end
    X = newX;
end

% best final
f = zeros(pop,1);
for i=1:pop
    [routeIdx, routeXY] = decode_ranked(X(i,:), P, sXY, gXY, nVisit);
    f(i) = fitness_route(routeXY, routeIdx, valN, lambda);
end
[bestFit, bidx] = min(f);
[bestRouteIdx, bestPathXY] = decode_ranked(X(bidx,:), P, sXY, gXY, nVisit);
cumVal = cumulative_value(bestRouteIdx, valN);
end

% ---- rank-coded decode: take top-nVisit, then order by nearest neighbor ----
function [routeIdx, routeXY] = decode_ranked(w, P, sXY, gXY, nVisit)
K = size(P,1);
nVisit = min(nVisit, K);
[~,ord] = sort(w,'descend');
sel = unique(ord(1:nVisit),'stable');
pts = P(sel,:);
route = order_by_nearest([sXY; pts; gXY]);
% map back to indices: for cumulative value, we need indices of pts only
routeIdx = sel(order_by_nearest_indices(pts, sXY, gXY));
routeXY  = route;
end

function idx = order_by_nearest_indices(pts, sXY, gXY)
% Greedy: start at sXY, visit nearest, end at gXY via nearest insertion
n = size(pts,1);
if n==0, idx=[]; return; end
unv = 1:n;
path = []; cur = sXY;
while ~isempty(unv)
    [~,j] = min(sum((pts(unv,:)-cur).^2,2));
    path(end+1) = unv(j); %#ok<AGROW>
    cur = pts(unv(j),:);
    unv(j) = [];
end
% ensure last is closest to goal
idx = path;
end

function route = order_by_nearest(Pfull)
% Pfull includes s, pts..., g (already concatenated)
% We only reorder the inner points greedily
s = Pfull(1,:); g = Pfull(end,:);
pts = Pfull(2:end-1,:);
if isempty(pts), route = Pfull; return; end
n = size(pts,1);
unv = 1:n; path = [];
cur = s;
while ~isempty(unv)
    [~,j] = min(sum((pts(unv,:)-cur).^2,2));
    path(end+1) = unv(j); %#ok<AGROW>
    cur = pts(unv(j),:);
    unv(j) = [];
end
route = [s; pts(path,:); g];
end

% ---- fitness and cumulative value ----
function f = fitness_route(routeXY, routeIdx, valN, lambda)
if isempty(routeIdx)
    f = 1e9; return;
end
L = sum(sqrt(sum(diff(routeXY,1,1).^2,2)));
R = sum(valN(routeIdx));
f = L - lambda*R;
end

function c = cumulative_value(routeIdx, valN)
if isempty(routeIdx), c=0; return; end
c = cumsum(valN(routeIdx));
end

% ---- GA helpers ----
function k = tournament(f)
n = numel(f);
a = randi(n); b = randi(n);
if f(a) < f(b), k=a; else, k=b; end
end
function [c1,c2] = blend_crossover(p1,p2)
alpha = 0.5;
u = rand(size(p1));
c1 = alpha*u.*p1 + (1-alpha*u).*p2;
c2 = alpha*u.*p2 + (1-alpha*u).*p1;
end
function y = mutate_gauss(x, pm)
y = x;
mask = rand(size(x)) < pm;
y(mask) = y(mask) + 0.15*randn(sum(mask(:)),1);
end


%% =================== Geometry / indexing core ===================
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
        HE(idxHE).twin = 0; HE(idxHE).next = 0; HE(idxHE).prev = 0; HE(idxHE).face = f;
        loc(e) = idxHE;
        if isempty(DCEL.vertices(vA).halfEdge), DCEL.vertices(vA).halfEdge = idxHE; end
        k = edgekey_undirected(vA,vB);
        if isKey(edgeMap,k), edgeMap(k) = [edgeMap(k) idxHE]; else, edgeMap(k) = idxHE; end
    end
    HE(loc(1)).next = loc(2); HE(loc(2)).next = loc(3); HE(loc(3)).next = loc(1);
    HE(loc(1)).prev = loc(3); HE(loc(2)).prev = loc(1); HE(loc(3)).prev = loc(2);
    faceHE(f) = loc(1);
end
keysList = keys(edgeMap);
for kk=1:numel(keysList)
    k = keysList{kk}; heList = edgeMap(k);
    if numel(heList)==2
        a = heList(1); b = heList(2); HE(a).twin = b; HE(b).twin = a;
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

function [Q, rectXY] = random_fov_rect(W,H,side)
cx = 20 + rand*(W-40); cy = 20 + rand*(H-40);
half = side/2;
xs = max(1, cx-half); ys = max(1, cy-half);
xe = min(W, cx+half); ye = min(H, cy+half);
Q = [xs ys; xe ys; xe ye; xs ye; xs ys];
rectXY = [xs ys xe ye]; %#ok<NASGU>
end

function [quad_data, leaves] = build_qdcel_wrapper(triangles, vertices, triValues, leaf_pix, params, W, H)
if nargin < 6, W=512; H=512; end
[~, quad_data, small_squares_data] = quadtree_split_v2( ...
    1,1,W,H, triangles, vertices, triValues, [], 'Q_', leaf_pix, [], [], params);
leaves = small_squares_data; %#ok<NASGU>
end

% -------- Quadtree (your original split) --------
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

width = x_end - x_start + 1; height = y_end - y_start + 1;
relevant_triangles = find_relevant_triangles_bbox(triangles, vertices, x_start, y_start, x_end, y_end);

if isempty(params.altitude), tau = params.tau0;
else, tau = params.tau0 * max(1, params.altitude/params.z0)^(-params.alpha);
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

current_node = struct('code', code, 'x_start',x_start,'y_start',y_start, ...
    'x_end',x_end,'y_end',y_end,'value',quad_value,'depth',params.depth);
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

% ---- FoV overlap (your robust version) ----
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
        if abs(overlap_area - aTri) <= 1e-6 * aTri || abs(overlap_area - aRect) <= 1e-6 * aRect
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
                if local_segments_touch_or_intersect(p1,p2,q1,q2), touch = true; break; end
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
        overlap_area = 0; relation = 'disjoint'; return;
    else
        overlap_area = polyarea(ix, iy);
        aTri  = max(polyarea(tri_coords(:,1), tri_coords(:,2)),  tol);
        aRect = max(polyarea(rect_coords(:,1), rect_coords(:,2)), tol);
        if abs(overlap_area - aTri) <= 1e-6 * aTri || abs(overlap_area - aRect) <= 1e-6 * aRect
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
o1 = local_orientation(p1,p2,q1); o2 = local_orientation(p1,p2,q2);
o3 = local_orientation(q1,q2,p1); o4 = local_orientation(q1,q2,p2);
if (o1 ~= o2) && (o3 ~= o4), tf = true; return; end
if o1 == 0 && local_on_segment(p1,q1,p2), tf = true; return; end
if o2 == 0 && local_on_segment(p1,q2,p2), tf = true; return; end
if o3 == 0 && local_on_segment(q1,p1,q2), tf = true; return; end
if o4 == 0 && local_on_segment(q1,p2,q2), tf = true; return; end
end
function val = local_orientation(a,b,c)
v = (b(2)-a(2))*(c(1)-b(1)) - (b(1)-a(1))*(c(2)-b(2));
if abs(v) < 1e-12, val = 0; elseif v > 0, val = 1; else, val = 2; end
end
function tf = local_on_segment(a,b,c)
tf = (min(a(1),c(1)) - 1e-12 <= b(1)) && (b(1) <= max(a(1),c(1)) + 1e-12) && ...
     (min(a(2),c(2)) - 1e-12 <= b(2)) && (b(2) <= max(a(2),c(2)) + 1e-12);
end

% ---- MBR & rect helpers ----
function rects = tri_mbrs(V, T)
n = size(T,1); rects = zeros(n,4);
for i=1:n
    tri = V(T(i,:),:);
    rects(i,:) = [min(tri(:,1)) min(tri(:,2)) max(tri(:,1)) max(tri(:,2))];
end
end
function tf = poly_rect_intersects(poly, rect)
% rect=[xmin ymin xmax ymax]
if any(isnan(poly(:))), tf=false; return; end
px = [rect(1) rect(3) rect(3) rect(1) rect(1)];
py = [rect(2) rect(2) rect(4) rect(4) rect(2)];
[ix,iy] = polybool('intersection', poly(:,1), poly(:,2), px(:), py(:));
tf = ~isempty(ix) && ~isempty(iy);
end
function idx = find_by_bbox(T,V,rect)
% rect=[xmin ymin xmax ymax]
n = size(T,1); keep=false(n,1);
for i=1:n
    tri = V(T(i,:),:);
    bx = [min(tri(:,1)) min(tri(:,2)) max(tri(:,1)) max(tri(:,2))];
    if ~(bx(3) < rect(1) || rect(3) < bx(1) || bx(4) < rect(2) || rect(4) < bx(2))
        keep(i)=true;
    end
end
idx = find(keep);
end

% ---- RTree (STR bulk-load) ----
function RT = rtree_build(rects, M)
N = size(rects,1);
idx = (1:N).';
centx = (rects(:,1)+rects(:,3))/2; [~, px] = sort(centx);
S = ceil(N / M);
Sx = ceil(sqrt(S));
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
        leaves{end+1} = node;
    end
end
RT = rtree_pack(leaves, M);
end
function node = rtree_pack(children, M)
if numel(children) == 1, node = children{1}; return; end
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
        parents{end+1} = n; 
    end
end
node = rtree_pack(parents, M);
end
function outIdx = rtree_query(node, rect)
outIdx = [];
if ~mbr_intersect(node.mbr, rect), return; end
if node.isLeaf
    keep = mbr_intersect_many(node.rects, rect);
    outIdx = node.idx(keep);
else
    for i=1:numel(node.children)
        outIdx = [outIdx; rtree_query(node.children{i}, rect)]; 
    end
end
end
function tf = mbr_intersect(a,b)
tf = ~(a(3) < b(1) || b(3) < a(1) || a(4) < b(2) || b(4) < a(2));
end
function keep = mbr_intersect_many(R, b)
keep = ~(R(:,3) < b(1) | b(3) < R(:,1) | R(:,4) < b(2) | b(4) < R(:,2));
end

% ---- misc ----
function s = normalize01(x)
if isempty(x), s=x; return; end
mn = min(x); mx = max(x);
if mx>mn, s = (x-mn)/(mx-mn); else, s = zeros(size(x)); end
end
function [p1,p2] = farthest_pair(P)
D = pdist2(P,P);
[~,idx] = max(D(:)); [i,j] = ind2sub(size(D), idx);
p1 = P(i,:); p2 = P(j,:);
end
function set_isi_style()
set(groot,'defaultAxesFontName','Times New Roman');
set(groot,'defaultAxesFontSize',11);
set(groot,'defaultLineLineWidth',1.5);
set(groot,'defaultAxesLineWidth',1.0);
set(groot,'defaultAxesGridAlpha',0.15);
end
