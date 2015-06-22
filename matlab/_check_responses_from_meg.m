%% identify files
files = dir([data_path 'fif/*.fif']);
files = files(SelFileOrder{sbj_number}); % put files in order

%% preprocessing of each block
f = 1;
%% Trial definition
cfg = [];
cfg.dataset             = fullfile([data_path 'fif'],files(f).name);
cfg.headerformat        = 'neuromag_mne';
cfg.dataformat          = 'neuromag_mne';
%cfg.trialfun           = 'ft_trialfun_general';
cfg.trialdef.eventtype  = 'STI101';
cfg.trialdef.eventvalue = 1:25;
cfg.trialdef.prestim    = .800; % number, latency in seconds (optional);
cfg.trialdef.poststim   = 3.000; % number, latency in seconds (optional);
cfg                     = ft_definetrial(cfg);

%% Read in the data
%cfg.channel     = {'MEG','MEGREF'};
cfg.continuous  = 'yes';
data            = ft_preprocessing(cfg);
channel = find(ismember(data.label, 'STI101'));
channel = find(ismember(data.label, 'STI101'));
sti=cell2mat(cellfun(@(x) x(channel,:)', data.trial, 'uniformoutput', false));
imagesc(sti')