import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function predict(payload) {
  const response = await axios.post(`${API_URL}/predict`, payload)
  return response.data
}
