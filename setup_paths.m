
switch 'niccolo_ubuntu'
    case 'D145'
        cd('/media/niccolo/Yupi/Paris/scripts');
        path = '/media/niccolo/Yupi/Paris/';
        addpath('/media/niccolo/Yupi/Paris/toolboxes/JR_toolbox/');
        addpath('/media/niccolo/Yupi/Paris/toolboxes/export_fig/');
        addpath('/media/niccolo/Yupi/Paris/toolboxes/fieldtrip-20140123/'); ft_defaults;
        addpath('/media/niccolo/Yupi/Paris/toolboxes/circular_stats/');
        addpath('/media/niccolo/Yupi/Paris/toolboxes/statistics/');
        addpath('/media/niccolo/Yupi/myfunctions');
        addpath('/media/niccolo/Yupi/Paris/toolboxes/resampling_statistical_toolkit');
    case 'D145_wind'
        cd('F:/Paris/scripts');
        path = 'F:/Paris/';
        addpath('F:/Paris/toolboxes/JR_toolbox/');
        addpath('F:/Paris/toolboxes/export_fig/');
        addpath('F:/Paris/toolboxes/fieldtrip-20140123/'); ft_defaults;
        addpath('F:/Paris/toolboxes/circular_stats/');
        addpath('F:/Paris/toolboxes/statistics/');
        addpath('F:/myfunctions');
        addpath('F:/Paris/toolboxes/resampling_statistical_toolkit');
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
        path = '/media/My Passport/Paris/';
        addpath(path);
        addpath('/home/niccolo/Dropbox/DOCUP/toolboxes/JR_toolbox/');
        addpath('/home/niccolo/Dropbox/DOCUP/toolboxes/export_fig/');
        %addpath('/home/niccolo/Dropbox/DOCUP/toolboxes/fieldtrip-20130225/'); ft_defaults;
        addpath('/home/niccolo/Dropbox/DOCUP/toolboxes/fieldtrip-20141030/'); ft_defaults;
        addpath('/home/niccolo/Dropbox/DOCUP/toolboxes/circular_stats/');
        
end
