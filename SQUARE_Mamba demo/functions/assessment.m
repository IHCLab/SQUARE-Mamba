function RESULTS = assessment(gt, prediction)

    MAE = mean(abs(gt - prediction));
    RMSE = sqrt(mean((gt - prediction).^2));
    SS_res = sum((gt - prediction).^2);        
    SS_tot = sum((gt - mean(gt)).^2);           
    R2 = 1 - (SS_res / SS_tot);  

	% Store results:
	RESULTS.MAE = MAE;
	RESULTS.RMSE = RMSE;
	RESULTS.R2 = R2;

end