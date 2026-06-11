clc,clear
close all


load  urban
[no_lines,no_rows,no_bands] = size(M);
obs = reshape(image,no_lines*no_rows,no_bands).';
[~,score,latent,tsquare] = pca(obs');
energy = cumsum(latent)./sum(latent);
index = find(energy >= .9999);
pc = index(1);
data = score(:,1:pc);

D = pdist(data, 'euclidean');
dist = squareform(D);

ND=size(dist,2);
N=ND*(ND-1)/2;

percent=2.0;
fprintf('average percentage of neighbours (hard coded): %5.6f\n', percent);

position=round(N*percent/100);
sda=sort(D);
dc=sda(position);

fprintf('Computing Rho with gaussian kernel of radius: %12.6f\n', dc);

for i=1:ND
  rho(i)=0.;
end
%
% Gaussian kernel
%
for i=1:ND-1
  for j=i+1:ND
     rho(i)=rho(i)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
     rho(j)=rho(j)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));%计算局部密度
  end
end

maxd=max(max(dist));

[rho_sorted,ordrho]=sort(rho,'descend');
delta(ordrho(1))=-1.;
nneigh(ordrho(1))=0;

for ii=2:ND
   delta(ordrho(ii))=maxd;
   for jj=1:ii-1
     if(dist(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))
        delta(ordrho(ii))=dist(ordrho(ii),ordrho(jj));
        nneigh(ordrho(ii))=ordrho(jj);
     end
   end
end
delta(ordrho(1))=max(delta(:));

rho = (rho-min(rho))./(max(rho)-min(rho));
delta = (delta-min(delta))./(max(delta)-min(delta));

ind = find(delta>=0.05);
rho1 = rho(ind);
delta1 = delta(ind);
gamma1 = rho1.*delta1.^2;
[a,b] = sort(gamma1,'descend');
figure;semilogy(a(1:20),'o')

set(gca,'Fontsize',12,'Fontname','times new roman')
figure;tt=plot(rho(:),delta(:),'o','MarkerSize',5,'MarkerFaceColor','k','MarkerEdgeColor','k');
xlabel ('\rho','Fontsize',16)
ylabel ('\delta','Fontsize',16)

eta = 0.1
for i= 1:length(a)
    if abs(log10(a(i+1)/a(i+2)))<eta && abs(log10(a(i+2)/a(i+3)))<eta
        num_ind = i
        break
    end
end
if length(b)>num_ind
    ind=ind(b(1:num_ind));
end


NCLUST=0;
for i=1:ND
  cl(i)=-1;
end
for i=1:length(ind)
     NCLUST=NCLUST+1;
     cl(ind(i))=NCLUST;
     icl(NCLUST)=ind(i);
end
fprintf('NUMBER OF CLUSTERS: %i \n', NCLUST);
disp('Performing assignation')

%assignation
for i=1:ND
  if (cl(ordrho(i))==-1)
    cl(ordrho(i))=cl(nneigh(ordrho(i)));
  end
end
%halo
for i=1:ND
  halo(i)=cl(i);
end
if (NCLUST>1)
  for i=1:NCLUST
    bord_rho(i)=0.;
  end
  for i=1:ND-1
    for j=i+1:ND
      if ((cl(i)~=cl(j))&& (dist(i,j)<=dc))
        rho_aver=(rho(i)+rho(j))/2.;
        if (rho_aver>bord_rho(cl(i))) 
          bord_rho(cl(i))=rho_aver;
        end
        if (rho_aver>bord_rho(cl(j))) 
          bord_rho(cl(j))=rho_aver;
        end
      end
    end
  end
  for i=1:ND
    if (rho(i)<bord_rho(cl(i)))
      halo(i)=0;
    end
  end
end
for i=1:NCLUST
  nc=0;
  nh=0;
  for j=1:ND
    if (cl(j)==i) 
      nc=nc+1;
    end
    if (halo(j)==i) 
      nh=nh+1;
    end
  end
  fprintf('CLUSTER: %i CENTER: %i ELEMENTS: %i CORE: %i HALO: %i \n', i,icl(i),nc,nh,nc-nh);
end

cmap=colormap;
for i=1:NCLUST
   ic=int8((i*64.)/(NCLUST*1.));
   hold on
   plot(rho(icl(i)),delta(icl(i)),'o','MarkerSize',8,'MarkerFaceColor',cmap(ic,:),'MarkerEdgeColor',cmap(ic,:));
end

thres = 80*100*0.005;
oldicl = icl;
da = obs(:,icl);
set_small = [];
for i = 1:NCLUST;
    num = numel(find(cl==i));
    if num>=thres
        continue;
    end
    set_small = [set_small, i];
end

for i = 1:numel(set_small)
    index = set_small(i);
    da1 = obs(:,icl(index));
    dis = sum((repmat(da1,1,NCLUST)-da).^2);
    dis(set_small) = 1e4;
    [~,b] = sort(dis,'ascend');
    ind = find(cl==index);
    cl(ind) = b(1);
end
icl(set_small) = [];


ss = numel(icl);
P = 20
B = [];

ind1= [];
for i = 1:numel(icl)
    newcl = icl(i);
    oldcl = find(oldicl == newcl);
    index = find(cl==oldcl);
    rho1 = rho(index);
    delta1 = delta(index);
    gamma = rho1.*delta1.^2;
    [~,b]=sort(gamma,'descend');
    if isempty(intersect(index(b(1:P)),icl(i)))
        ind_sel = [icl(i), index(b(1:P-1))];
    else
        ind_sel = index(b(1:P));
    end
    ind1 = [ind1,ind_sel];
    B = [B, obs(:,ind_sel)];
end
image = M;
mask = groundtruth;

A =B;
save('urban1',"image","mask","A","-v7.3");
