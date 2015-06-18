
switch 'niccolo_ubuntu'
    case 'D145'
        cd('/home/niccolo/Dropbox/DOCUP/scripts');
        path = '/media/niccolo/Yupi/Paris/';
        im_path = '/media/niccolo/Yupi/Paris/images_acrossSubjs';
        addpath(path);
        addpath('/home/niccolo/Dropbox/DOCUP/toolboxes/JR_toolbox/');
        addpath('/home/niccolo/Dropbox/DOCUP/toolboxes/export_fig/');
        addpath('/home/niccolo/Dropbox/DOCUP/toolboxes/fieldtrip-20141030/'); ft_defaults;
        addpath('/home/niccolo/Dropbox/DOCUP/toolboxes/circular_stats/');
        addpath('/home/niccolo/Dropbox/DOCUP/toolboxes/statistics/');
        addpath(genpath('/home/niccolo/Dropbox/Oxford/myfunctions/'));
%         addpath('/media/niccolo/Yupi/myfunctions');
%         addpath('/media/niccolo/Yupi/Paris/toolboxes/resampling_statistical_toolkit');
    case 'D145_wind'
        cd('C:/Users/npescetelli/Dropbox/DOCUP/scripts');
        path = 'F:/Paris/';
        addpath(genpath('C:/Users/npescetelli/Dropbox/DOCUP/toolboxes/JR_toolbox'));
        addpath('C:/Users/npescetelli/Dropbox/DOCUP/toolboxes/circular_stats/');
        addpath(genpath('C:/Users/npescetelli/Dropbox/Oxford/myfunctions'));
    case 'jr'
        cd('/media/DATA/Pro/Projects/Paris/Orientation/Niccolo/script/201301');
        path = '/media/My Passport/Paris 16-01-2014/';
        addpath('/media/DATA/Pro/Toolbox/JR_toolbox/');
        addpath('/media/DATA/Pro/Toolbox/export_fig/');
        addpath('/media/DATA/Pro/Toolbox/fieldtrip/fieldtrip-20130225/'); ft_defaults;
        addpath('/media/DATA/Pro/Toolbox/circular_stats/');
    case 'imen'
        cd('/home/imen/Desktop/Backup_ICM/Projects_ICM/2014_meg_invisible_orientations/scripts/data_processing/');
        path = '/media/My Passport/Paris 16-01-2014/';
        addpath('/home/imen/Desktop/Backup_ICM/Projects_ICM/toolboxes/JR_toolbox/');
        addpath('/home/imen/Desktop/Backup_ICM/Projects_ICM/toolboxes/export_fig/');
        addpath('/home/imen/Desktop/Backup_ICM/Projects_ICM/toolboxes/fieldtrip-20130319/'); ft_defaults;
        addpath('/home/imen/Desktop/Backup_ICM/Projects_ICM/toolboxes/circular_stats/');
        addpath('/home/imen/Desktop/Backup_ICM/Projects_ICM/toolboxes/statistics/');
    case 'niccolo_ubuntu'
        cd('/home/niccolo/Dropbox/DOCUP/scripts');
        path = '/media/Paris/';
        addpath(path);
        addpath('/home/niccolo/Dropbox/DOCUP/toolboxes/JR_toolbox/');
        addpath('/home/niccolo/Dropbox/DOCUP/toolboxes/export_fig/');
        addpath('/home/niccolo/Dropbox/DOCUP/toolboxes/fieldtrip-20141030/'); ft_defaults;
        addpath('/home/niccolo/Dropbox/DOCUP/toolboxes/circular_stats/');
        
end
