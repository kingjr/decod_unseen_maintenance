function data = loadBin2mat(filename)
% data = loadBin2mat(filename)
% retrieve fieldtrip structure from binary (.dat) + header (.mat)

load(filename,'data');
trials = binload([filename(1:(end-4)) '.dat'], data.Xdim);
for t = size(trials,1):-1:1
    data.trial{t} = squeeze(trials(t,:,:));
    % clear data along the way
    trials(t,:,:) = [];
end
return