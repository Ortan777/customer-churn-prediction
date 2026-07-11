import { useState } from 'react'
import { predict } from './api'

const defaultPayload = {
  gender: 'Female',
  SeniorCitizen: 0,
  Partner: 'No',
  Dependents: 'No',
  tenure: 12,
  PhoneService: 'Yes',
  MultipleLines: 'No',
  InternetService: 'Fiber optic',
  OnlineSecurity: 'No',
  OnlineBackup: 'No',
  DeviceProtection: 'No',
  TechSupport: 'No',
  StreamingTV: 'No',
  StreamingMovies: 'No',
  Contract: 'Month-to-month',
  PaperlessBilling: 'Yes',
  PaymentMethod: 'Electronic check',
  MonthlyCharges: 70.35,
  TotalCharges: 844.2,
}

export default function App() {
  const [formData, setFormData] = useState(defaultPayload)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleChange = (event) => {
    const { name, value } = event.target
    setFormData((prev) => ({
      ...prev,
      [name]: name === 'MonthlyCharges' || name === 'TotalCharges' || name === 'tenure' || name === 'SeniorCitizen'
        ? Number(value)
        : value,
    }))
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const response = await predict(formData)
      setResult(response)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  // Client-side risk helper
  const getRiskDetails = (prob) => {
    if (prob >= 0.7) return { label: 'HIGH', class: 'active-high', percent: (prob * 100).toFixed(1) }
    if (prob >= 0.4) return { label: 'MEDIUM', class: 'active-medium', percent: (prob * 100).toFixed(1) }
    return { label: 'LOW', class: 'active-low', percent: (prob * 100).toFixed(1) }
  }

  const risk = result ? getRiskDetails(result.probability) : null

  return (
    <div className="page">
      <header>
        <div className="header-badge">
          <span className="dot"></span>
          XGBoost Model Active
        </div>
        <h1>Customer Churn Predictor</h1>
        <p>Predict customer churn risk using our trained machine learning model.</p>
      </header>

      <main>
        <section className="panel">
          <h2>Customer Details</h2>
          <form onSubmit={handleSubmit}>
            <div className="grid">
              {/* Demographics */}
              <label>
                Gender
                <select name="gender" value={formData.gender} onChange={handleChange}>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </label>

              <label>
                Senior Citizen
                <select name="SeniorCitizen" value={formData.SeniorCitizen} onChange={handleChange}>
                  <option value="0">No</option>
                  <option value="1">Yes</option>
                </select>
              </label>

              <label>
                Partner
                <select name="Partner" value={formData.Partner} onChange={handleChange}>
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </label>

              <label>
                Dependents
                <select name="Dependents" value={formData.Dependents} onChange={handleChange}>
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </label>

              {/* Tenure & Charges */}
              <label>
                Tenure (months)
                <input type="number" name="tenure" min="0" max="72" value={formData.tenure} onChange={handleChange} />
              </label>

              <label>
                Monthly Charges ($)
                <input type="number" name="MonthlyCharges" step="0.01" min="0" value={formData.MonthlyCharges} onChange={handleChange} />
              </label>

              <label>
                Total Charges ($)
                <input type="number" name="TotalCharges" step="0.01" min="0" value={formData.TotalCharges} onChange={handleChange} />
              </label>

              {/* Services */}
              <label>
                Phone Service
                <select name="PhoneService" value={formData.PhoneService} onChange={handleChange}>
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </label>

              <label>
                Multiple Lines
                <select name="MultipleLines" value={formData.MultipleLines} onChange={handleChange}>
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No phone service">No phone service</option>
                </select>
              </label>

              <label>
                Internet Service
                <select name="InternetService" value={formData.InternetService} onChange={handleChange}>
                  <option value="DSL">DSL</option>
                  <option value="Fiber optic">Fiber optic</option>
                  <option value="No">No</option>
                </select>
              </label>

              <label>
                Online Security
                <select name="OnlineSecurity" value={formData.OnlineSecurity} onChange={handleChange}>
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </label>

              <label>
                Online Backup
                <select name="OnlineBackup" value={formData.OnlineBackup} onChange={handleChange}>
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </label>

              <label>
                Device Protection
                <select name="DeviceProtection" value={formData.DeviceProtection} onChange={handleChange}>
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </label>

              <label>
                Tech Support
                <select name="TechSupport" value={formData.TechSupport} onChange={handleChange}>
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </label>

              <label>
                Streaming TV
                <select name="StreamingTV" value={formData.StreamingTV} onChange={handleChange}>
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </label>

              <label>
                Streaming Movies
                <select name="StreamingMovies" value={formData.StreamingMovies} onChange={handleChange}>
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </label>

              {/* Contract & Billing */}
              <label>
                Contract
                <select name="Contract" value={formData.Contract} onChange={handleChange}>
                  <option value="Month-to-month">Month-to-month</option>
                  <option value="One year">One year</option>
                  <option value="Two year">Two year</option>
                </select>
              </label>

              <label>
                Paperless Billing
                <select name="PaperlessBilling" value={formData.PaperlessBilling} onChange={handleChange}>
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </label>

              <label>
                Payment Method
                <select name="PaymentMethod" value={formData.PaymentMethod} onChange={handleChange}>
                  <option value="Electronic check">Electronic check</option>
                  <option value="Mailed check">Mailed check</option>
                  <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                  <option value="Credit card (automatic)">Credit card (automatic)</option>
                </select>
              </label>
            </div>
            <button type="submit" disabled={loading}>
              {loading ? 'Processing...' : 'Predict Churn'}
            </button>
          </form>
          {error && <div className="error">{error}</div>}
        </section>

        {result && risk && (
          <section className="panel result">
            <h2>Prediction Result</h2>
            <div className="result-header">
              <span className={`result-badge ${result.prediction === 'Yes' ? 'churn-yes' : 'churn-no'}`}>
                {result.prediction === 'Yes' ? 'WILL CHURN' : 'WILL STAY'}
              </span>
              <p>
                {result.prediction === 'Yes' 
                  ? 'This customer is likely to leave.' 
                  : 'This customer is likely to stay.'}
              </p>
            </div>

            <div className="prob-container">
              <div className="prob-header">
                <span className="prob-label">Churn Probability</span>
                <span className="prob-value">{risk.percent}%</span>
              </div>
              <div className="prob-bar-bg">
                <div 
                  className={`prob-bar-fill ${result.probability >= 0.7 ? 'high' : result.probability >= 0.4 ? 'medium' : 'low'}`}
                  style={{ width: `${risk.percent}%` }}
                ></div>
              </div>
            </div>

            <div className="risk-level">
              <div className={`risk-item ${risk.class === 'active-low' ? 'active-low' : ''}`}>
                <div className="risk-icon">🟢</div>
                <div className="risk-name">Low Risk</div>
              </div>
              <div className={`risk-item ${risk.class === 'active-medium' ? 'active-medium' : ''}`}>
                <div className="risk-icon">🟡</div>
                <div className="risk-name">Medium Risk</div>
              </div>
              <div className={`risk-item ${risk.class === 'active-high' ? 'active-high' : ''}`}>
                <div className="risk-icon">🔴</div>
                <div className="risk-name">High Risk</div>
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  )
}
