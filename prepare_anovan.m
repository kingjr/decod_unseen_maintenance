function [Y, GROUP] = prepare_anovan(Y)
% [Y, GROUP] = prepare_anovan(Y)
% vectorize n dimension matrix and output the corresponding group to be
% used with anovan.

% store Yatrix size
sizes = size(Y);
% flatten Yatrix down
Y = Y(:);
% build group vector
n = length(sizes);
GROUP = {};
for dim = 1:n
    g = zeros(sizes);
    for line = 1:sizes(dim)
        eval(['g(' repmat(':,', [1, dim-1]) 'line' repmat(',:', [1, n-dim]) ')=line;'])
    end
    GROUP{dim} = g(:); 
end
