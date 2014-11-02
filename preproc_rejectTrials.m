%% Summary stats for artifacts rejection
%http://fieldtrip.fcdonders.nl/tutorial/visual_artifact_rejection

%% select channels of interest
selchan = @(c,s) find(cell2mat(cellfun(@(x) ~isempty(strfind(x,s)),c,'uniformoutput', false))==1);

% %% data browser to visually inspect data
% cfg             = [];
% cfg.viewmdode   = 'vertical';
% cfg.continuous  = 'no';
% 
% cfg = ft_databrowser(cfg,data)

%% Reject by visual inspection
cfg          = [];
cfg.method   = 'summary';
cfg.alim     = 1e-12;
cfg.megscale = 1;
cfg.eogscale = 5e-8;
cfg.channel  = selchan(data.label,'MEG');
dummy        = ft_rejectvisual(cfg,data);

removed_trials{sbj_number} = input('Which trials have been rejected?');
removed_chans(sbj_number)  = find(~ismember([1:306],selchan(dummy.label,'MEG'))); % returns bad MEG channels

%% possibly use databroswer again to double check
%
%
%% remove rejected trials from MEG behaviour
trials=trials(~ismember([1:length(trials)],removed_trials{sbj_number}));
data_clean = dummy;

