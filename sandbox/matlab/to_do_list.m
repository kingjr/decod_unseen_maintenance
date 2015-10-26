Before:
- find some papers (or maybe ask Valentin?) where they decoded from time-frequency in order to select few bands of interest

Paper:
- structuring introduction


MEG preproc:
- remove bad channels
- redo preprocessing on response-locked segmentation
- (possibly remove 2 subjects that can be decoded from eye movements)
- redo preprocesing on time frequency for all subjects and selected frequencies


Univariate Analysis:
- find a way to analyse results from topos, at the moment ANOVAS on concatenated data crash for RAM saturation
- plot final figures for topos
- do topos on univariate time frequencies and find some suitable stats


Multivariate Anlaysis:
- Rerun decoding on generalization matrix only (traditional decoding can then be inferred from diagonal) both for SVC and SVR. Currently working on it
- manage to decode accuracy
- transform main recursive scripts (like defineCOntrast) into functions. Partially done
- try all visualizations (eg v-stat, r-stat, p-values, prediction errors...) but select only a few for the final paper
- redo decoding on time frequency analysis

Overall:
- clean scripts and comment as much as possible
- have a document where all the analyses we do are specified, and put some images of the figure and the reference to the script that generated them. This will help you structure the systematic analyses. Possible use some kind of tree like structure.
- what's the story we want to convey with the time frequency analysis?
- understand how github works and trying not to mess up with it



