function save_var(var,path,sbj_initials,type) 
eval([inputname(1) '=var;'])
save([path 'data/' sbj_initials '/' type '/' sbj_initials '_' type '_' inputname(1) '.mat'],inputname(1), 'time', 'labels')