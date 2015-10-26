function out = my_mean(in)
if size(in,1)>10
    out = squeeze(trimmean(in,90,'round',1));
elseif size(in,1)==0
    out = NaN(size(in,2), size(in,3));
else
    out = squeeze(mean(in,1));
end
    