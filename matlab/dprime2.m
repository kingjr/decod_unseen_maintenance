function [dPrime, lnbeta, C] = dprime2(hit,miss,fa,cr)
% [dPrime, beta, C] = dprime2(hit,miss,fa,cr)

% DPRIME -- Signal-detection theory sensitivity measure.
%
% d = dprime(pHit,pFA, PresVector)
% [dPrime,beta,C] = dprime(pHit,pFA)
%
% PHIT and PFA are numerical arrays of the same shape.
% PHIT is the proportion of "Hits": P(Yes|Signal)
% PFA is the proportion of "False Alarms": P(Yes|Noise)
% PRESENTT is the number of Signal Present Trials e.g. length(find(signal==1))
% ABSENTT is the number of Signal Absent Trials e.g. length(find(signal==0))
%
% All numbers involved must be between 0 and 1.
% The function calculates the d-prime measure for each pair.
% The criterion values BETA and C can also be requested.
% Requires MATLAB's Statistical Toolbox.
%
% NK BW 207 Feb. 17 2010
% adapted by JRKING january 2013, jeanremi.king at gmail dot com
% ======================================================================= %

PresentT =(hit+miss);
AbsentT = (fa+cr);
pHit = hit/PresentT;
pFA = fa/AbsentT;

%--Error checking
if pHit == 1 % if 100% Hits
% if e.g. 50 signal present trials Hits is between 98-100%
% make Hits 1-(1/N2) [1-(1/(2*50)) = .99]
% N is the number of signal present trials
pHit = 1-(1/(2*PresentT)); %
end
if pFA == 0 % if 0% FA
% if e.g. 50 signal absent trials FA is between 0-2%
% make FA 1/N2 [1/(2*50) = .01]
% N is the number of signal absent trials
pFA = 1/(2*AbsentT);
end

%-- Convert to Z scores
zHit = norminv(pHit) ;
zFA = norminv(pFA) ;

%-- Calculate d-prime
dPrime = zHit - zFA ;
%beta = exp((zFA^2 - zHit^2)/2);
lnbeta = (zFA^2 - zHit^2)/2;
C = -.5 * (zHit + zFA);
