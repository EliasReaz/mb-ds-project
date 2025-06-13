
### Summary for Non-Technical Client

- The goal was to **predict how many people would use the ferry each day**.

- The original model didn’t perform well in predicting daily ticket counts, likely because the data has both weekly and monthly seasonality, as well as occasional sharp increases in visitors.

- I explored a few forecasting methods. One of them, called **Prophet** (developed by Meta), worked well for regular patterns like monthly or yearly trends, but it struggled with sudden changes.

- Another option, a machine learning model called **XGBoost**, was better at picking up both weekly and yearly patterns. However, it still missed some unexpected spikes.

- So I used a **two-step approach**, combining **XGBoost with an error correction step**. This method behaves more like a smart decision-maker. It learns usual patterns (like busy weekends and summer months), then makes extra adjustments for unusual days (like long weekends or holidays).

- I also built a second model to forecast **ticket sales** using the same framework of XGBoost along with error correction, which helps understand customer demand ahead of time.

- To make everything clear, I added graphs to show how the model works and which patterns it pays attention to - so you can trust the forecast and explain it easily to others.
