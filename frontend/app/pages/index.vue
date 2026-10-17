<template>
  <div class="w-full flex-1 flex flex-col items-center">
    <!-- 1. Hero — Headline above the video -->
    <HeroSection />

    <!-- 2. Stats Ticker (Between video and features) -->
    <section class="w-full py-10 px-4 border-y border-[#d5ccbe]/50">
      <div class="max-w-5xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
        <div v-for="stat in stats" :key="stat.label" class="flex flex-col items-center gap-1">
          <div class="text-3xl md:text-4xl font-black text-slate-900 tracking-tight">{{ stat.value }}</div>
          <div class="text-[11px] font-bold text-[#8b7d6b] uppercase tracking-widest">{{ stat.label }}</div>
        </div>
      </div>
    </section>

    <!-- 3. Features Grid -->
    <section id="features-section" class="w-full py-16 md:py-20 px-4">
      <div class="max-w-5xl mx-auto">
        <div class="text-center mb-12">
          <p class="text-xs font-black text-green-600 uppercase tracking-[0.2em] mb-2">Why BeejRakshak?</p>
          <h3 class="text-3xl md:text-4xl font-black text-slate-900 tracking-tight">Intelligence at Every Step</h3>
          <p class="mt-3 text-sm text-[#8b7d6b] max-w-lg mx-auto leading-relaxed">
            From field to mandi — our ML pipeline analyses price trends, transport costs, and perishability to maximize your profit.
          </p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-5">
          <div v-for="feature in features" :key="feature.title" 
            class="bg-white/70 border border-[#dcd1c1] rounded-2xl p-6 transition-all duration-300 hover:shadow-xl hover:shadow-green-600/5 hover:-translate-y-1 hover:border-green-500/30 group">
            <div class="w-12 h-12 bg-gradient-to-br from-green-50 to-green-100 border border-green-200/50 rounded-xl flex items-center justify-center text-2xl mb-4 group-hover:scale-110 transition-transform duration-300">
              {{ feature.icon }}
            </div>
            <h4 class="text-base font-extrabold text-slate-900 mb-1.5">{{ feature.title }}</h4>
            <p class="text-sm text-[#8b7d6b] leading-relaxed">{{ feature.desc }}</p>
          </div>
        </div>
      </div>
    </section>

    <!-- 4. ML Pipeline Visual Strip -->
    <section class="w-full py-14 px-4 bg-gradient-to-b from-transparent via-[#f5f0e6]/60 to-transparent">
      <div class="max-w-4xl mx-auto">
        <div class="text-center mb-10">
          <p class="text-xs font-black text-green-600 uppercase tracking-[0.2em] mb-2">Under the Hood</p>
          <h3 class="text-3xl md:text-4xl font-black text-slate-900 tracking-tight">Our 3-Stage ML Pipeline</h3>
        </div>

        <div class="flex flex-col md:flex-row items-center md:items-stretch gap-4">
          <div v-for="(stage, idx) in pipeline" :key="stage.name" 
            class="flex-1 bg-white border border-[#dcd1c1] rounded-2xl p-5 text-center relative">
            <div class="w-8 h-8 bg-green-600 text-white rounded-full flex items-center justify-center font-black text-sm mx-auto mb-3 shadow-md shadow-green-600/20">
              {{ idx + 1 }}
            </div>
            <h5 class="text-sm font-extrabold text-slate-900 mb-1">{{ stage.name }}</h5>
            <p class="text-xs text-[#8b7d6b] leading-relaxed">{{ stage.desc }}</p>
            <!-- Arrow connector -->
            <div v-if="idx < pipeline.length - 1" class="hidden md:block absolute -right-3 top-1/2 -translate-y-1/2 text-green-500 text-xl font-bold z-10">→</div>
          </div>
        </div>
      </div>
    </section>

    <!-- 5. Main Form + Results Section -->
    <section id="form-section" class="w-full flex-1 flex flex-col items-center py-16 px-4 pb-20">
      <div class="text-center mb-8">
        <p class="text-xs font-black text-green-600 uppercase tracking-[0.2em] mb-2">Try It Now</p>
        <h3 class="text-3xl md:text-4xl font-black text-slate-900 tracking-tight">Get Your Best Mandi</h3>
        <p class="mt-3 text-sm text-[#8b7d6b] max-w-md mx-auto leading-relaxed">
          Select your crop, enter the quantity, and our engine will calculate the optimal selling strategy across all mandis.
        </p>
      </div>

      <FormCard @result="onResult" />

      <div v-if="resultData" class="w-full flex justify-center mt-8">
        <ResultCard :data="resultData" />
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import HeroSection from '~/components/HeroSection.vue'
import FormCard from '~/components/FormCard.vue'
import ResultCard from '~/components/ResultCard.vue'

const resultData = ref<any>(null)

function onResult(data: any) {
  resultData.value = data
}

const stats = [
  { value: '~1.62%', label: 'MAPE Error' },
  { value: '7+', label: 'Gujarat Mandis' },
  { value: '3', label: 'Crop Models' },
  { value: '2 yrs', label: 'Historical Data' },
]

const features = [
  { icon: '📈', title: 'XGBoost Forecasting', desc: 'Fast, secure XGBoost Regressors forecast precise market prices 3-7 days into the future.' },
  { icon: '🚛', title: 'Dynamic Fleet Logic', desc: 'Automatically selects LCVs or Heavy Trucks based on your payload and scales multi-trip costs.' },
  { icon: '🧊', title: 'Perishability Engine', desc: 'Exponential decay models per crop — so you know exactly how much you lose waiting in a truck.' },
  { icon: '💰', title: 'Net Profit > Gross', desc: 'We calculate real financial take-home by subtracting exact fuel, tolls, wear-and-tear, and labor.' },
  { icon: '🏪', title: 'Spatial Arbitrage', desc: 'Compares every mandi in Gujarat to find the one with the highest true financial return.' },
  { icon: '⏳', title: 'Temporal Arbitrage', desc: 'Automatically advises whether to sell now or wait, factoring in storage and spoilage costs.' },
]

const pipeline = [
  { name: 'XGBoost Price Prediction', desc: 'Predict exactly what the market clearing price will be based on 2-year trends and momentum.' },
  { name: 'Logistics Analytics', desc: 'Calculate dynamic transport scaling based on payload, perishability, and distance.' },
  { name: 'Profit Maximization', desc: 'Mathematical evaluation to output the optimal Net Profit, ignoring deceptive high gross prices.' },
]

</script>
