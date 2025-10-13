Target date range: 2024-02-14 00:00:00 to 2025-10-14 00:00:00
Window size: 608 days

XRP-GBP, 
XMR-GBP, 
LTC-GBP, 
XLM-GBP

norm: mean, embedding: manual norm + copy

Returning best val model from epoch 15
-----------------------------------------                                                                                                                                                
Best validation model: eval_train metrics
-----------------------------------------
- total: 517968.0
- rmse: 0.0008320596787849664
- mse: 0.003600709118051914
- mae: 0.03374821308795307
- informer_rmse: 0.041524757538817424
- informer_mse: 0.003595669725474886
- informer_mae: 0.03374369015928204
----------------------------------
Best validation model: val metrics
----------------------------------
- total: 85680.0
- rmse: 0.0012383305023308674
- mse: 0.005242152542682812
- mae: 0.051141560200104415
- informer_rmse: 0.06156282422931066
- informer_mse: 0.005231681457579336
- informer_mae: 0.05110892719988312
-----------------------------------
Best validation model: test metrics
-----------------------------------
- total: 84768.0
- rmse: 0.001343816142598158
- mse: 0.006087323007335209
- mae: 0.05626918505434272
- informer_rmse: 0.06867306473265801
- informer_mse: 0.006368438052179824
- informer_mae: 0.05729055015503296

Weight decay changed from 1e-4 to 1e-3:

Returning best val model from epoch 15
-----------------------------------------                                                                                                                                                
Best validation model: eval_train metrics
-----------------------------------------
- total: 517968.0
- rmse: 0.0008320779803825832
- mse: 0.0036011745164706704
- mae: 0.03374914680475714
- informer_rmse: 0.04152578252475136
- informer_mse: 0.0035961356285478625
- informer_mae: 0.03374464794464365
----------------------------------
Best validation model: val metrics
----------------------------------
- total: 85680.0
- rmse: 0.001238208921516643
- mse: 0.005241125801892953
- mae: 0.051136037548717
- informer_rmse: 0.06155672533038471
- informer_mse: 0.005230642122374515
- informer_mae: 0.05110328315225031
-----------------------------------
Best validation model: test metrics
-----------------------------------
- total: 84816.0
- rmse: 0.0013448777121038982
- mse: 0.006097476929426193
- mae: 0.05630463045782648
- informer_rmse: 0.06866313507115203
- informer_mse: 0.006365270929693777
- informer_mae: 0.057272162421473434

50% reverse training augmentation makes things worse:

Returning best val model from epoch 30
-----------------------------------------                                                                                                                                                
Best validation model: eval_train metrics
-----------------------------------------
- total: 517968.0
- rmse: 0.0008370492560635077
- mse: 0.0033786775227271316
- mae: 0.03408087945362247
- informer_rmse: 0.0421424484536184
- informer_mse: 0.003375481521057338
- informer_mae: 0.034084232289140176
----------------------------------
Best validation model: val metrics
----------------------------------
- total: 85680.0
- rmse: 0.0012645845027530894
- mse: 0.0056175471555674665
- mae: 0.05225822317833994
- informer_rmse: 0.06292540992477111
- informer_mse: 0.005614735597711322
- informer_mae: 0.05229920310167862
-----------------------------------
Best validation model: test metrics
-----------------------------------
- total: 84864.0
- rmse: 0.0014229863067333156
- mse: 0.006636934867620378
- mae: 0.05941880670787702
- informer_rmse: 0.07202629844791122
- informer_mse: 0.006829473858358272
- informer_mae: 0.060177889020581334

Back to linear time and 1e-4 Weight decay. Added Time embeddings:
Returning best val model from epoch 16
-----------------------------------------                                                                                                                                                
Best validation model: eval_train metrics
-----------------------------------------
- total: 517968.0
- rmse: 0.0008217734757509222
- mse: 0.0034605196526948014
- mae: 0.033185618561817246
- informer_rmse: 0.040891228940822845
- informer_mse: 0.003454670320552288
- informer_mae: 0.03316870223890615
----------------------------------
Best validation model: val metrics
----------------------------------
- total: 85680.0
- rmse: 0.0012754307616324651
- mse: 0.005541554193536774
- mae: 0.05262584935605916
- informer_rmse: 0.06347815462920282
- informer_mse: 0.005532602076917621
- informer_mae: 0.05261960419427071
-----------------------------------
Best validation model: test metrics
-----------------------------------
- total: 84912.0
- rmse: 0.0013261297747523676
- mse: 0.00579259978140144
- mae: 0.05585286022917304
- informer_rmse: 0.06654363671051604
- informer_mse: 0.005882693827775906
- informer_mae: 0.056312241624774675

embeddings interleave

Returning best val model from epoch 16
-----------------------------------------
Best validation model: eval_train metrics
-----------------------------------------
- total: 517968.0
- rmse: 0.0010977829722646462
- mse: 0.005702611313637934
- mae: 0.045248482345382146
- informer_rmse: 0.05399376688492016
- informer_mse: 0.005693046240535214
- informer_mae: 0.04522435086332923
----------------------------------
Best validation model: val metrics
----------------------------------
- total: 85680.0
- rmse: 0.0016756131612007915
- mse: 0.011729151059209911
- mae: 0.07350250379044135
- informer_rmse: 0.08159703152653362
- informer_mse: 0.011694496666937735
- informer_mae: 0.07339771461140897
-----------------------------------
Best validation model: test metrics
-----------------------------------
- total: 84960.0
- rmse: 0.0014637347198531677
- mse: 0.007602054914792828
- mae: 0.06221956335667612
- informer_rmse: 0.07217232320856835
- informer_mse: 0.007697120225722236
- informer_mae: 0.06268786273098417

with fourier attn block and channel time emb 48 channels



