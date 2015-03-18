function [trial_proportion predict_angle radius] = recombine_SVR_prediction(results_x,results_y,angles,res)
%[trial_proportion predict_angle radius] = decode_reg2angle(results_x,results_y,angles,res)
% this function takes the two outputs of an SVR to transform them into
% predicted angles and predicted radi;
% the resolution (res) specify the bin width for the 2D histogram
warning('obsolete function name! This function is now called decode_computeSVRerror.m')