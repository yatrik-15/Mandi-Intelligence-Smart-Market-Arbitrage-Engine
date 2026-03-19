# Mandi Arbitrage Engine (Mandi Intelligence Module)

**Smart logistics AI for BeejRakshak** - Calculate true net profit for farmers by factoring in transportation costs, wait times, and perishability.

## 🎯 Overview

This module helps farmers make data-driven decisions about where to sell their produce. Instead of just showing market prices, it calculates **true net profit** by accounting for:

- 💰 Market price per kg
- 🚚 Transportation costs (₹15/km for small truck)
- ⏱️ Wait time costs (5% perishability deduction for high crowd levels)

## 📐 The Algorithm

The core formula used for net profit calculation:

```
Net Profit = (Price × Qty) - (Distance × FuelCost) - (WaitTime × PerishabilityFactor)
```

**Constants:**
- `FUEL_COST_PER_KM = ₹15` (assumes small truck)
- `PERISHABILITY_DEDUCTION = 5%` (applied for "High" crowd levels)

## 🏗️ Project Structure

```
mandi_intelligence/
├── data/
│   └── mandi_data.json          # Mock data for 5 mandis around Gandhinagar
├── core/
│   ├── profit_calculator.py     # Core calculation engine
│   └── insight_generator.py     # Human-readable recommendations
├── api/
│   └── main.py                  # FastAPI application
├── scripts/
│   └── visualize.py             # Console visualization tool
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd mandi_intelligence
pip install -r requirements.txt
```

### 2. Run the Visualization Script

See the algorithm in action with a console table:

```bash
python scripts/visualize.py
```

This will print comparison tables for Cotton, Wheat, and Onion showing how **net profit differs from gross earnings**.

### 3. Start the API Server

```bash
uvicorn api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

## 📡 API Endpoints

### **GET** `/get_best_mandi`

Get optimal mandi recommendations for a crop.

**Query Parameters:**
- `crop` (required): Crop name - `"Cotton"`, `"Wheat"`, or `"Onion"`
- `quantity` (required): Quantity to sell in kg (must be > 0)
- `current_location` (optional): Current location (defaults to "Gandhinagar")

**Example Request:**
```
GET http://localhost:8000/get_best_mandi?crop=Cotton&quantity=1000
```

**Example Response:**
```json
{
  "crop": "Cotton",
  "quantity": 1000,
  "recommendations": [
    {
      "mandi_id": "MANDI_E",
      "mandi_name": "Ahmedabad APMC",
      "location": "Ahmedabad",
      "distance_km": 40,
      "price_per_kg": 58,
      "crowd_level": "Low",
      "gross_earnings": 58000,
      "transport_cost": 600,
      "perishability_cost": 0,
      "net_profit": 57400
    },
    {
      "mandi_id": "MANDI_D",
      "mandi_name": "Visnagar Wholesale Market",
      "location": "Visnagar",
      "distance_km": 35,
      "price_per_kg": 56,
      "crowd_level": "Medium",
      "gross_earnings": 56000,
      "transport_cost": 525,
      "perishability_cost": 0,
      "net_profit": 55475
    }
  ],
  "top_recommendation": {
    "mandi": "Ahmedabad APMC",
    "insight": "Go to Ahmedabad APMC (40km). Even though it is 35km further than Sardar Patel Mandi, the price is ₹8/kg higher, so you will make ₹7,625 more profit after fuel costs."
  }
}
```

### **GET** `/health`
Health check endpoint.

### **GET** `/mandis`
List all available mandis with their details.

### **GET** `/docs`
Interactive API documentation (Swagger UI).

## 🗺️ Mock Data

The module includes data for **5 mandis** around Gandhinagar:

| Mandi | Distance | Cotton Price | Wheat Price | Onion Price | Crowd |
|-------|----------|--------------|-------------|-------------|-------|
| Sardar Patel Mandi | 5 km | ₹50/kg | ₹25/kg | ₹30/kg | Low |
| Kalol Agricultural Market | 15 km | ₹52/kg | ₹26/kg | ₹32/kg | High |
| Mehsana Grain Market | 25 km | ₹54/kg | ₹28/kg | ₹35/kg | Low |
| Visnagar Wholesale Market | 35 km | ₹56/kg | ₹30/kg | ₹38/kg | Medium |
| **Ahmedabad APMC** | **40 km** | **₹58/kg** | **₹33/kg** | **₹40/kg** | **Low** |

**Key Design:** The furthest mandi (Ahmedabad APMC) has significantly higher prices to demonstrate arbitrage opportunities.

## 💡 Example Use Cases

### Use Case 1: Cotton Farmer with 1000 kg

**Naive approach:** Go to nearest mandi (5km)
- Gross earnings: ₹50,000
- Transport cost: ₹75
- **Net profit: ₹49,925**

**Smart approach:** Go to Ahmedabad APMC (40km)
- Gross earnings: ₹58,000
- Transport cost: ₹600
- **Net profit: ₹57,400** ✅

**Result:** ₹7,475 more profit by traveling 35km further!

### Use Case 2: Understanding Crowd Impact

If Ahmedabad APMC had "High" crowd level:
- Gross earnings: ₹58,000
- Transport cost: ₹600
- Perishability cost: ₹2,900 (5% of gross)
- **Net profit: ₹54,500**

Still profitable, but the high crowd reduces profit by ₹2,900.

## 🛠️ Development

### Running Tests

The visualization script serves as a functional test:

```bash
python scripts/visualize.py
```

It should demonstrate that:
1. ✅ Further mandis can be more profitable
2. ✅ Transportation costs are correctly deducted
3. ✅ Perishability costs apply to high-crowd mandis
4. ✅ Rankings are sorted by net profit (not price)

### Adding More Mandis

Edit `data/mandi_data.json` to add more mandis. Each mandi needs:
```json
{
  "id": "MANDI_X",
  "name": "Market Name",
  "location": "City",
  "distance_km": 20,
  "crowd_level": "Low|Medium|High",
  "crop_prices": {
    "Cotton": 55,
    "Wheat": 28,
    "Onion": 35
  }
}
```

## 🔮 Future Enhancements

- [ ] Real-time eNAM API integration
- [ ] Dynamic fuel cost based on current prices
- [ ] Multi-modal transport (truck vs. train)
- [ ] Weather-based perishability adjustment
- [ ] Historical price trends
- [ ] Route optimization for multiple mandis

## 📄 License

Part of the BeejRakshak AgriTech Platform.

---

**Built with ❤️ for farmers** | Phase 1 & 4 Implementation | Mandi Intelligence Module
