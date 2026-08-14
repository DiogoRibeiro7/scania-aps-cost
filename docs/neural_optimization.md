# Neural optimization study

The PyTorch MLP is intentionally configurable so optimizer behaviour can be studied rather than hidden behind one default training recipe.

For parameters \(\theta\), SGD follows

\[
\theta_{t+1}=\theta_t-\eta_t\nabla_\theta L_t.
\]

Adam maintains first- and second-moment estimates,

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
\]

and AdamW decouples weight decay from the adaptive gradient update. The implementation exposes all three optimizers.

The experimental controls are:

- learning rate;
- constant, cosine, or plateau-driven scheduling;
- batch size;
- gradient clipping;
- weight decay;
- dropout;
- batch normalization;
- architecture width/depth;
- early stopping;
- class-weighted binary cross entropy;
- focal loss.

The estimator records epoch-level train and validation loss in `history_`, so convergence and overfitting can be inspected directly.
