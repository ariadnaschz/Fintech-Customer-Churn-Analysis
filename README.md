# 🏦 Predictive Analytics: Detección de Fuga de Clientes (Churn) y Capital en Riesgo en Fintech

## 📊 Business Case
En el entorno Fintech, el costo de adquisición de clientes (CAC) es elevado, pero el costo de retener a un cliente valioso es mínimo si se interviene a tiempo. Este proyecto no solo clasifica a los usuarios con riesgo de abandono, sino que cuantifica el **Impacto Financiero** de dicha fuga para priorizar los esfuerzos comerciales.

## 🛠️ Metodología y Stack Tecnológico
1. **Ingeniería de Datos (Python / Pandas):** - Limpieza de datos y Feature Engineering estratégico.
   - Creación de variables de negocio como `Is_Zero_Balance` (para detectar "cuentas zombie") y `Tenure_Age_Ratio`.
2. **Machine Learning (Scikit-Learn):** - Entrenamiento de un modelo `RandomForestClassifier`.
   - Implementación de *Cross-Validation* y balanceo de clases (`class_weight='balanced'`) para combatir el 36% de inactividad oculta en el dataset.
3. **Business Intelligence (Power BI / DAX):** - Diseño de un dashboard ejecutivo estructurado bajo la regla de 3 niveles (KPIs, Tendencias, Detalle Accionable).

## 🚀 Impacto de Negocio Descubierto
Al cruzar las predicciones del modelo (Probabilidad de abandono > 75%) con los saldos de las cuentas, se detectó una exposición de capital de **$90.31 Millones** en riesgo de fuga inminente. 

El modelo generó una lista de "Hot Leads" (Cuentas Críticas), permitiendo al equipo de retención ejecutar llamadas preventivas enfocadas exclusivamente en el segmento de mayor LTV (Lifetime Value), optimizando así el presupuesto operativo.

## 🖥️ Dashboard
![Dashboard](https://github.com/ariadnaschz/Fintech-Customer-Churn-Analysis/blob/main/Reports/figures/dashboard_preview.png?raw=true)