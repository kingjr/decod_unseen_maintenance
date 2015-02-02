%% subjects_demographics
cd('F:\Paris\scripts');
clear all
close all
clc

%% libraries & toolboxes
setup_paths

%% All subjects
setup_subjects;


%% demographics 
for s = 1 : length(SubjectsList)
    sbj_initials = SubjectsList{s};
    data_path = [path 'data/' sbj_initials '/'] ;
    load([data_path 'behavior/' sbj_initials '.mat'],'subject')
    
    age(s) = subject.age;
    rhand(s) = subject.right_handed;
    male(s) = subject.male;
    
end

mean(age);std(age);sum(male);sum(rhand);

