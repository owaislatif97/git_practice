import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

def build_pinn_model(input_dim, hidden_layers=[64, 64, 32]):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    for units in hidden_layers:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

def physics_loss(y_true, y_pred, wind_speed,
                 cut_in=3.0, rated=12.0, cut_out=25.0, rated_power=1.0):
    """
    Physics-based loss using a simplified turbine power curve.
    
    Regions:
    - v < cut-in:         P = 0
    - cut-in ≤ v < rated: P ∝ v³ normalized to rated power
    - rated ≤ v ≤ cut-out: P = rated_power
    - v > cut-out:        P = 0
    """
    # Clamp wind speeds
    ws = tf.squeeze(wind_speed)
    
    # Define piecewise theoretical power
    zero = tf.zeros_like(ws)
    rated_p = tf.ones_like(ws) * rated_power
    v_norm = tf.pow((ws - cut_in) / (rated - cut_in + 1e-6), 3.0) * rated_power

    # Conditions
    cond_region1 = ws < cut_in
    cond_region2 = tf.logical_and(ws >= cut_in, ws < rated)
    cond_region3 = tf.logical_and(ws >= rated, ws <= cut_out)
    cond_region4 = ws > cut_out

    theoretical_power = tf.where(cond_region1, zero,
                          tf.where(cond_region2, v_norm,
                          tf.where(cond_region3, rated_p, zero)))

    # Mean squared error between predicted and theoretical
    return tf.reduce_mean(tf.square(tf.squeeze(y_pred) - theoretical_power))


def train_pinn_model(model, X_train, y_train, wind_speed_col_idx,
                     physics_weight=0.1, learning_rate=0.001,
                     epochs=100, batch_size=32, validation_data=None, verbose=1):
    """
    Train the Physics-Informed Neural Network with improved physics loss.
    """
    # Ensure inputs are numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    wind_speed = X_train[:, wind_speed_col_idx:wind_speed_col_idx+1]

    # Optimizer
    optimizer = Adam(learning_rate=learning_rate)

    # History for tracking
    history = {'loss': [], 'val_loss': [] if validation_data else None}

    # tf.data.Dataset pipeline
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, wind_speed))
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Preprocess validation data if given
    if validation_data:
        X_val, y_val = validation_data
        X_val = np.array(X_val).astype(np.float32)
        y_val = np.array(y_val).astype(np.float32)

    # Physics-based loss using realistic wind turbine power curve
    def physics_loss(y_pred, wind_speed,
                     cut_in=3.0, rated=12.0, cut_out=25.0, rated_power=1.0):
        ws = tf.squeeze(wind_speed)
        zero = tf.zeros_like(ws)
        rated_p = tf.ones_like(ws) * rated_power
        v_norm = tf.pow((ws - cut_in) / (rated - cut_in + 1e-6), 3.0) * rated_power

        cond_region1 = ws < cut_in
        cond_region2 = tf.logical_and(ws >= cut_in, ws < rated)
        cond_region3 = tf.logical_and(ws >= rated, ws <= cut_out)
        cond_region4 = ws > cut_out

        theoretical_power = tf.where(cond_region1, zero,
                              tf.where(cond_region2, v_norm,
                              tf.where(cond_region3, rated_p, zero)))

        return tf.reduce_mean(tf.square(tf.squeeze(y_pred) - theoretical_power))

    # Training step
    @tf.function
    def train_step(X_batch, y_batch, ws_batch):
        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)

            # Losses
            mse = tf.reduce_mean(tf.square(y_batch - y_pred))
            phys = physics_loss(y_pred, ws_batch)
            total = mse + physics_weight * phys

        grads = tape.gradient(total, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total

    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0

        for X_b, y_b, ws_b in train_dataset:
            loss = train_step(tf.cast(X_b, tf.float32),
                              tf.cast(tf.expand_dims(y_b, axis=-1), tf.float32),
                              tf.cast(ws_b, tf.float32))
            total_loss += loss
            batch_count += 1

        avg_loss = total_loss / batch_count
        history['loss'].append(avg_loss.numpy())

        # Validation
        if validation_data:
            val_pred = model.predict(X_val, verbose=0)
            val_loss = tf.reduce_mean(tf.square(y_val - val_pred)).numpy()
            history['val_loss'].append(val_loss)

        # Print
        if verbose and (epoch + 1) % 10 == 0:
            if validation_data:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model, history


def predict_pinn(model, X_test):
    return model.predict(X_test, verbose=0).flatten()
