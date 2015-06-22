function handle = imagesc_meg(time,data,pos,clim,factor,type)
if nargin <5 || isempty(factor), factor = 10; end
if nargin <6 || isempty(type), type = 'vertical'; end
%% sort channels
switch type
    case 'vertical'
        [z chans] = sort(pos(:,2), 'descend');
        chans(chans>306) = [];
    case 'vertical_back_split'
        [z chans] = sort(pos(:,2), 'descend');
        left = chans(pos(chans,1)<0);
        right = chans(pos(chans,1)>=0);
        chans = cat(1,left,right(end:-1:1));
        chans(chans>306) = [];
    case 'vertical_back_split_mag'
        [z chans] = sort(pos(:,2), 'descend');
        left = chans(pos(chans,1)<0);
        right = chans(pos(chans,1)>=0);
        chans = cat(1,left,right(end:-1:1));
        chans(setdiff(1:306,3:3:306)) = [];
        chans(chans>306) = [];
end

%% normalize magnetometers
data(3:3:306,:) = data(3:3:306,:).*factor;

%% plot
if exist('clim', 'var') && ~isempty(clim)
    handle = imagesc(time,[],data(chans,:),clim);
else
    handle = imagesc(time,[],data(chans,:));
end