% COMPUTE_MDS_SET_ERROR
%
% E = compute_mds_set_error(d,x,ind,[i]) 
%
% d	distance matrix for sets
% x	2-D or 3-D vectors (mapped) [for set elements]
% ind   index mapping x elements to sets (in order to map to d)
% i     compute only for the ith set 
% E	Error functional
%
% TEST SCRIPTS (set N):
% d = rand(N); x = 2*rand(2*N,2)-1; ind = [1:N,1:N]; compute_mds_set_error(d,x,ind);
% for i = 1:N; a(i) = compute_mds_set_error(d,x,ind,i); end; sum(a)
% SPEED Test:
% d = rand(N); x = 2*rand(5*N,2)-1; ind = [1:N,1:N,1:N,1:N,1:N]; compute_mds_set_error(d,x,ind);
% for i = 1:N; a(i) = compute_mds_set_error(d,x,ind,i); end; sum(a)

% AGP, Oct 19, 2011, TUC

function E = compute_mds_set_error(d,x,ind,i)

if (nargin == 4) %%% faster depending on average set cardinality (x avg.set.cardiality/2 faster)
  N = size(d,1);
  K = size(x,1);
  dim = size(x,2);
  E = 0;
  valid_x_ind = find(ind ~=0);
  iind = find(ind == i);
  if (dim == 2)
    for k = 1:length(valid_x_ind)
      j = valid_x_ind(k); 
      tmp(j) = min(sqrt((x(iind,1)-x(j,1)).^2 + (x(iind,2)-x(j,2)).^2));
    end
    for j = 1:N
      jind = find(ind == j);
      E = E + (min(tmp(jind))-d(i,j)).^2;
    end
  elseif (dim == 3)
    for k = 1:length(valid_x_ind)
      j = valid_x_ind(k); 
      tmp(j) = min(sqrt((x(iind,1)-x(j,1)).^2 + (x(iind,2)-x(j,2)).^2 + (x(iind,3)-x(j,3)).^2));
    end
    for j = 1:N
      jind = find(ind == j);
      E = E + (min(tmp(jind))-d(i,j)).^2;
    end
  end
else  %%% slow but only runs once 
  N = size(d,1);
  K = size(x,1);
  E = 0;
  dext = zeros(K,K);
  valid_x_ind = find(ind ~=0);
  for k = 1:length(valid_x_ind) 
    i = valid_x_ind(k);
    for l = 1:length(valid_x_ind)
      j = valid_x_ind(l);
      dext(i,j) = sqrt(sum((x(i,:)-x(j,:)).^2));
    end
  end
  for i = 1:N 
    iind = find(ind == i);
    for j = 1:N
      jind = find(ind == j);
      E = E + (min(min(dext(iind,jind)))-d(i,j)).^2;
    end
  end
end







% COMPUTE_MDS_ERROR
%
% E = compute_mds_error(d,x,[i]) 
%
% d	distance matrix
% x	2-D or 3-D vectors (mapped)
% i     compute only for the ith element
% E	Error functional
%
% TEST SCRIPTS (set N):
% d = rand(N); x = 2*rand(N,2)-1; compute_mds_error(d,x);
% for i = 1:N; a(i) = compute_mds_error(d,x,i); end; sum(a)

% AGP, Oct 19, 2011, TUC

function E = compute_mds_error(d,x,i)

if (nargin == 3)
  y = x;
  for k = 1:size(x,2);
    y(:,k) = (y(:,k)-x(i,k));
  end
  E = sum((sqrt(sum(y'.^2))-d(i,:)).^2); 
else  %%% slow but only runs once 
  N = size(d,1);
  E = 0;
  for i = 1:N 
    for j = 1:N
      E = E + (sqrt(sum((x(i,:)-x(j,:)).^2))-d(i,j)).^2;
    end
  end
end




% MDS Given a matrix of distances d(i,j) (or similarities)
%     compute the set of 2D vectors x(i,1:2) at minimize
%     the functional E = sum_ij (Euclidean_distance(x(i,:), x(j,:)) - d(i,j))^2
%
%  x = mds(d,dim)
%
%  d 	NxN distance matrix
%  x    Nx2 vectors
%  dim  2 or 3

% AGP, Oct 19, 2011, TUC

function x = mds(d,dim)



if (nargin == 1)
  dim = 2;
end

if (size(d,1) ~= size(d,2))
 error('Input distance matrix d should be square!');
end
if (size(d,1) < 3)
 error('Matrix size should be at least 3');
end
if ((dim < 2) || (dim > 3))
 error('Dimension of space to project to should be either 2 or 3');
end


N = size(d,1);

%%% NORMALIZE 0-1
%%d = d/max(max(d));
%
%%% STEP 1: randomly assign x in 0-1, 0-1
x = rand(N,dim); %%% 2-D or 3-D

%%% STEP 2: compute error
for i = 1:N
  E(i) = compute_mds_error(d,x,i); 
end

%%% STEP 3: Iterate
v = cos(pi/4);
%% WARNING: 2-D solution!!!!
points_org = [0 1; -1 0; 1 0 ; 0 -1;]; %%% reduce computation time by 50%
                                       %%%% introduces some warping of space!?
%points_org = [0 1; -1 0; 1 0 ; 0 -1; v v; v -v; -v v; -v -v];
Np = length(points_org);
Enew = sum(E);
iter = 0;
while ((iter == 0) || (Eorg > Enew))
  iter = iter + 1;
  Eorg = Enew;
  step = max(0.01,0.1/(sqrt(iter)));
  points = step*points_org;
  for i = 1:N;
    %%% heuristic, place between the closest two neighbhours
    %%% does not work because it collapses all points in a small region 
    %[tmp,ord] = sort(d(i,:));  
    %for j = 1:dim
    %  x(i,j) = 0.33*(x(ord(1),j) + x(ord(2),j)+x(ord(3),j));
    %end 
    % heuristic 2 (similar to gradient descent ...)
    xorg(i,:) = x(i,:);
    %%%% IDEA: one could reduce computation time here by up to 50%
    %%%%       by selecting a random direction to pertube (i.e., 
    %%%%       looking @ only two points at each iteration)
    %%%% BUG: moving along the x-y dimensions only causes a slight
    %%%%      warping of the final map (see IDEA above to improve things)
    E(i) = compute_mds_error(d,x,i); %%% BUG have to recompute E(i)
    for j = 1:Np
      x(i,:) = xorg(i,:) + points(j,:);
      Eloc(j) = compute_mds_error(d,x,i); 
    end
    [tmp,ord] = min(Eloc);
    %if (tmp < E(i)) %%%% does not work if you don't move => local minima - you have to pertubate!
    x(i,:) = xorg(i,:) + points(ord,:);
    Enew = Enew - 2*(E(i)-Eloc(ord)); %%%% see bug report below
    % disp(['Iter: ', num2str(iter), ' i:', num2str(i), ' Enew: ', num2str(Enew),'  E:',num2str(compute_mds_error(d,x))]);
    E(i) = Eloc(ord);
    %else
    %  x(i,:) = xorg(i,:);
    %end
  end  
  %%%% WARNING: Although each E(i) is correct when computed
  %%%%          by the end of the iteration everything has moved
  %%%%          so Enew is off. One could recompute it at the
  %%%%          cost of increasing complexity by 20%
  %%%% BUG: computing Enew as sum  of E is wrong because the effect
  %%%%      of moving x_i has only been counted on E(i) not on the whole matrix (x2 effect)
  %%%% ACTUALLY: Enew - 2*diff(E(i))
  %Enew = sum(E);  %%% BUG
  %disp([num2str(iter),'(',num2str(step),'): ',num2str(Enew),'(',num2str(sum(E)),')']);
  disp([num2str(iter),'(',num2str(step),'): ',num2str(Enew)]);
  %plot(x(:,1),x(:,2),'x');
end


% MDS_SET Given a matrix of distances d(i,j) (or similarities)
%         compute the set of 2D vectors x(i,1:2) at minimize
%         the functional E = sum_ij (Euclidean_distance(x(i,:), x(j,:)) - d(i,j))^2
%         However, here we are searching for hidden sets of elements
%         thus |x| cardinality is larger than the dimension of d.
%         In general, K = |x| = average_set_cardinality x N 
%         The common sense set distance is used here namely
%         E  = sum_ij (min_{iind, jind} Eucl_dist(x(iind,:), x(jind,:)) - d(i,j))^2
%         where iind are the indexes of x vectors that belong to set i
%         and jind similarly to set j.
%
%  [x,ind] = mds_set(d,K,dim)
%
%  d 	NxN distance matrix
%  K	the total number of set elements 
%  x    Nx2 vectors
%  ind  mapping between vectors and sets
%  dim  2 or 3
%

% AGP, Oct 19, 2011, TUC

function [x,ind] = mds_set(d,K,dim)


if (nargin == 2)
  dim = 2;
end

if (size(d,1) ~= size(d,2))
 error('Input distance matrix d should be square!');
end
if (size(d,1) < 3)
 error('Matrix size should be at least 3');
end
if ((dim < 2) || (dim > 3))
 error('Dimension of space to project to should be either 2 or 3');
end
if (K <= size(d,1))
 error('Number of x vectors should be larger than matrix size');
end

N = size(d,1);


%%% NORMALIZE 0-1
%%d = d/max(max(d));
%
%%% STEP 1: randomly assign x in 0-1, 0-1
x = zeros(K,dim);
x(1:N,:) = mds(d,dim);
ind = zeros(K,1); %%% ind stores the mapping between x elements and sets
ind(1:N) = [1:N];

%%% STEP 2: compute error
for i = 1:N
  E(i) = compute_mds_set_error(d,x,ind,i); 
end

%%% STEP 3&4 : Split & Iterate
%% WARNING: 2-D solution!!!!
points_org = [0 1; -1 0; 1 0 ; 0 -1;]; %%% reduce computation time by 50%
                                       %%%% introduces some warping of space!?
v = cos(pi/4);
points_split_org = [0 1; 1 0 ; v v; v -v]; %%% no need to split in other directions - symmetric
Np = length(points_org);

%%% split the most promising point
%x
%for i = 1:10; a(i) = compute_mds_set_error(d,x,ind,i);end
%a
%sum(a)
Kcur = N;
while (Kcur < K)  %%% split to desires number of total elements
  step_split = 0.03;
  points_split = step_split*points_split_org;
  for i = 1:N; E(i) = compute_mds_set_error(d,x,ind,i); end; % to be on the safe side
  for i = 1:Kcur;
  %disp(['i: ',num2str(i)]);
     xorg(i,:) = x(i,:);
     ind(Kcur+1) = ind(i);
     for j = 1:Np
       % disp(['  j: ',num2str(j)]);
       x(Kcur+1,:) = xorg(i,:) + points_split(j,:);
       x(i,:) = xorg(i,:) - points_split(j,:);
       Eloc_split(j) = compute_mds_set_error(d,x,ind,ind(i)); 
       x(i,:) = xorg(i,:);  %%% reset
     end
     [tmp,ord] = min(Eloc_split);
     Esplit(i) = tmp - E(ind(i)); %%% the one that reduces it the most
     split_ord(i) = ord;
  end
  %Esplit
  [tmp,ord] = min(Esplit);
  disp(['Spliting ', num2str(ord)]);
  x(ord,:) = xorg(ord,:) - points_split(split_ord(ord),:);
  x(Kcur+1,:) = xorg(ord,:) + points_split(split_ord(ord),:);
  ind(Kcur+1) = ind(ord); %%% bug fix for splits > 10 !!! 
  Kcur = Kcur + 1;
  %x
  %for i = 1:10; a(i) = compute_mds_set_error(d,x,ind,i);end
  %a
  %sum(a)
  for i = 1:N; E(i) = compute_mds_set_error(d,x,ind,i); end; % to be on the safe side
  disp(['Error after split: ',num2str(sum(E))]);

  % iterate
  Enew = sum(E);
  iter = 0;
  while ((iter == 0) || (Eorg > Enew))
    iter = iter + 1;
    Eorg = Enew;
    step = 0.005; %%% fixed - because mds with variable step only needs to be run in intialization step
    points = step*points_org;
    points_split = step*points_split_org;
    for i = 1:Kcur;
      xorg(i,:) = x(i,:);
      E(ind(i)) = compute_mds_set_error(d,x,ind,ind(i)); %%% bug fix (see mds.m)
      for j = 1:Np
        x(i,:) = xorg(i,:) + points(j,:);
        Eloc(j) = compute_mds_set_error(d,x,ind,ind(i)); 
      end
      [tmp,ord] = min(Eloc);
      x(i,:) = xorg(i,:) + points(ord,:);
      Enew = Enew - 2*(E(ind(i)) - Eloc(ord)); %% bug fix (see mds.m)
      E(ind(i)) = Eloc(ord);
    end  
    disp([num2str(iter),'(',num2str(step),'): ',num2str(Enew)]);
    %plot(x(:,1),x(:,2),'x');
  end

end

ind = ind';


