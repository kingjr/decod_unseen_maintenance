function [x cfg] = plot_topo_meg(x,time,cfg)
% [x cfg] = plot_topo_meg(x,cfg)
% x = chans x time x trial

if nargin < 3, cfg = []; end
if ~isfield(cfg, 'avg'), cfg.avg =true; end
if ~isfield(cfg, 'neighbours'),
    load('nm306all_neighb.mat', 'neighbours');
    cfg.neighbours      = neighbours;
end

%% format data
load('nm306_template', 'data');
xft         = data;
xft.trial   = {};
xft.time    = {};

for ii = 1:size(x,3)
    xft.trial{ii,1}       = x(:,:,ii);%{cat(1,x,ones(391-306,size(x,2)))};
    xft.time{ii,1}        = time;
end
xft.label = xft.label(1:size(x,1));
cfg.planarmethod    = 'sincos';
xft                 = ft_combineplanar(cfg,xft);

%% average
if cfg.avg
    avg = ft_timelockanalysis([],xft); %% WHAT THE FUCK?? FT INCREASES THE TIME BY TWO??
    if length(avg.time)>length(time)
        avg.time = time;
        avg.avg = avg.avg(:,1:length(time));
    end
end

%% Prepare plot
if ~isfield(cfg,'colorbar'),    cfg.colorbar = 'no'; end
if ~isfield(cfg,'layout'),      cfg.layout = 'neuromag306cmb.lay'; end
if ~isfield(cfg,'comment'),     cfg.comment = 'no'; end
if ~isfield(cfg,'marker'),      cfg.marker = 'off'; end
if ~isfield(cfg,'style'),       cfg.style= 'straight'; end

%% out
if nargout == 0
    % silent plot
    evalc('ft_topoplotER(cfg,avg);');
else
    % export data
    if ~isfield(cfg,'format'), cfg.format = 'ft'; end
    switch cfg.format
        case 'ft'
            x = avg;
        case 'mat'
            x = reshape(cell2mat(avg.trial),[size(avg.trial{1}),length(avg.trial)]);
    end
end