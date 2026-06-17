"""
hab_predictor.py
-----------------
Standalone module defining HABPredictor.

WHY THIS FILE NEEDS TO EXIST SEPARATELY:
joblib/pickle only stores a reference to a class (its module path + name),
not the actual class code. If HABPredictor is defined inside a notebook
or a one-off training script (e.g. as "__main__.HABPredictor"), then
joblib.load() will fail in any OTHER script (like this dashboard) with:

    AttributeError: Can't get attribute 'HABPredictor' on <module '__main__'>

Fix: define the class once, here, import it both when you TRAIN/SAVE the
model and when you LOAD it elsewhere (e.g. app.py).

After dropping this file next to your training script, re-run:
    from hab_predictor import HABPredictor
    ... build hab_predictor object as before ...
    joblib.dump(hab_predictor, 'hab_model.pkl')

Then this dashboard's joblib.load('D:/algal4/hab_model.pkl') will work.
"""

import numpy as np
import pandas as pd


class HABPredictor:
    """
    Complete HAB (Harmful Algal Bloom) Prediction Pipeline.

    Bundles a trained CatBoost model with all preprocessing,
    feature engineering, and WHO-style risk classification logic.

    Usage:
        model = joblib.load('hab_model.pkl')
        result = model.predict(new_data)
    """

    def __init__(self, model, feature_names, medians, n_pressure_cols, p_pressure_cols,
                 clip_dict, r2, rmse, mae):
        self.model = model
        self.feature_names = feature_names
        self.medians = medians
        self.n_pressure_cols = n_pressure_cols
        self.p_pressure_cols = p_pressure_cols
        self.clip_dict = clip_dict
        self.r2 = r2
        self.rmse = rmse
        self.mae = mae

    def engineer_features(self, df):
        """Create all engineered features used in training."""
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)

        if 'NTL' in df.columns and 'PTL' in df.columns:
            df['NP_ratio'] = df['NTL'] / (df['PTL'] + 1e-6)
            df['NP_ratio'] = df['NP_ratio'].clip(0, 200)

        if 'NTL' in df.columns:
            df['log_NTL'] = np.log1p(df['NTL'])
        if 'PTL' in df.columns:
            df['log_PTL'] = np.log1p(df['PTL'])
        if 'lake_AREA_HA' in df.columns:
            df['log_lake_area'] = np.log1p(df['lake_AREA_HA'])
        if 'WSAREA_km2' in df.columns:
            df['log_ws_area'] = np.log1p(df['WSAREA_km2'])
        if 'Depth' in df.columns:
            df['log_Depth'] = np.log1p(df['Depth'])

        available_n_cols = [c for c in self.n_pressure_cols if c in df.columns]
        available_p_cols = [c for c in self.p_pressure_cols if c in df.columns]
        df['total_N_pressure'] = df[available_n_cols].sum(axis=1) if available_n_cols else 0
        df['total_P_pressure'] = df[available_p_cols].sum(axis=1) if available_p_cols else 0

        if 'Month' in df.columns:
            df['peak_bloom_month'] = df['Month'].isin([7, 8]).astype(int)
        if 'lake_AREA_HA' in df.columns:
            df['small_lake'] = (df['lake_AREA_HA'] < 20).astype(int)

        if 'WSAREA_km2' in df.columns and 'lake_AREA_HA' in df.columns:
            df['ws_lake_ratio'] = df['WSAREA_km2'] / (df['lake_AREA_HA'] / 100 + 0.001)
            df['ws_lake_ratio'] = np.log1p(df['ws_lake_ratio'].clip(0, 10000))

        if 'Tmean' in df.columns and 'Tmean_YrMean' in df.columns:
            df['temp_anomaly'] = df['Tmean'] - df['Tmean_YrMean']
        if 'Precip' in df.columns and 'Precip_YrMean' in df.columns:
            df['precip_anomaly'] = df['Precip'] - df['Precip_YrMean']
        if 'LST' in df.columns and 'LST_YrMean' in df.columns:
            df['LST_anomaly'] = df['LST'] - df['LST_YrMean']
        if 'NPP' in df.columns and 'NPP_YrMean' in df.columns:
            df['NPP_anomaly'] = df['NPP'] - df['NPP_YrMean']

        if 'BFIWs' in df.columns and 'NAPI' in df.columns:
            df['bfi_x_napi'] = df['BFIWs'] * df['NAPI']
        if 'BFIWs' in df.columns and 'nani' in df.columns:
            df['bfi_x_nani'] = df['BFIWs'] * df['nani']

        if 'Legacy' in df.columns and 'Total Input' in df.columns:
            df['legacy_x_p_input'] = df['Legacy'] * df['Total Input']

        return df

    def predict(self, new_data):
        """
        Predict HAB risk for new lake data.

        Returns a DataFrame with Predicted_log_chla, Predicted_chla_ugL,
        Risk_Level (1-4), Risk_Label.
        """
        df = self.engineer_features(new_data)

        for col in self.feature_names:
            if col not in df.columns:
                df[col] = self.medians.get(col, 0)

        df = df[self.feature_names]
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(self.medians)

        for col in self.clip_dict:
            if col in df.columns:
                low, high = self.clip_dict[col]
                df[col] = df[col].clip(low, high)

        log_chla = self.model.predict(df)
        chla = 10 ** log_chla

        risks, risk_labels = [], []
        for c in chla:
            if c < 10:
                risks.append(1); risk_labels.append('Safe (<10 µg/L)')
            elif c < 50:
                risks.append(2); risk_labels.append('Caution (10-50 µg/L)')
            elif c < 100:
                risks.append(3); risk_labels.append('Warning (50-100 µg/L)')
            else:
                risks.append(4); risk_labels.append('Danger (>100 µg/L)')

        return pd.DataFrame({
            'Predicted_log_chla': np.round(log_chla, 3),
            'Predicted_chla_ugL': np.round(chla, 1),
            'Risk_Level': risks,
            'Risk_Label': risk_labels
        })

    def get_info(self):
        print("\n" + "=" * 50)
        print("HAB PREDICTION MODEL - Performance Summary")
        print("=" * 50)
        print(f"  Model Type    : CatBoost Regressor")
        print(f"  Test R²       : {self.r2:.4f}")
        print(f"  Test RMSE     : {self.rmse:.4f} (log scale)")
        print(f"  Test MAE      : {self.mae:.4f} (log scale)")
        print(f"  Total Features: {len(self.feature_names)}")
        print(f"  Risk Levels   : 4 (Safe/Caution/Warning/Danger)")
        print("=" * 50)
