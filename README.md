# Audio antispoofing contest

https://boosters.pro/championship/idrnd_antispoof/overview

## Models

Final score: 0.21 Equal error rate

Accuracy score on validation dataset: 0.983

Voting ensemble models: XGB, Conv1d-net(using raw 3s wav's), Conv2d-net(using MFCC).

Best single model: Conv1-net with LSTM layers acc: 0.994

## Data Preprocessing

Mel frequency cepstral coefficients, Fast Fourier Transform, raw wav's using padding/trim. 
