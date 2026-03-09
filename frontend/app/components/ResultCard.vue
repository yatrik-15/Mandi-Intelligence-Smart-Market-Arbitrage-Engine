<template>
  <div class="w-full max-w-[640px] mt-8 mb-16">
    
    <div class="bg-amber-100 border border-amber-300 rounded-xl p-4 text-sm text-amber-800 mb-6 leading-relaxed flex items-start gap-3 animate-[fadeUp_0.3s_ease]">
      <span class="text-xl">💡</span>
      <span>{{ data.summary }}</span>
    </div>

    <!-- BEST RECOMMENDATION HERO -->
    <div class="relative mb-8 animate-[fadeUp_0.35s_ease] group">
      <!-- glowing animated border -->
      <div class="absolute -inset-[2px] rounded-[1.4rem] bg-gradient-to-br from-green-500 via-[#4ade80] to-green-600 z-0 opacity-70 blur-0 animate-[borderPulse_3s_ease-in-out_infinite]"></div>
      
      <div class="relative z-10 bg-gradient-to-br from-[#f8f5eb] to-[#eedfbd] rounded-[1.3rem] p-8 overflow-hidden border border-[#d9c59a] shadow-lg">
        <!-- inner glow -->
        <div class="absolute -top-10 -right-10 w-48 h-48 rounded-full bg-white/40 blur-3xl pointer-events-none"></div>

        <div class="inline-flex items-center gap-2 bg-gradient-to-r from-green-600 to-green-500 text-white text-xs font-black tracking-widest uppercase rounded-full px-4 py-1.5 mb-4 shadow-[0_2px_12px_rgba(34,197,94,0.3)]">
          🏆 #1 Best Mandi
        </div>

        <h3 class="text-4xl md:text-5xl font-black leading-none text-slate-900 drop-shadow-sm mb-2">
          {{ best.mandi_name }}
        </h3>
        
        <div class="text-sm text-slate-600 mb-6">
          {{ best.distance_km.toFixed(0) }} km away &nbsp;·&nbsp; Sell <strong>{{ data.quantity.toLocaleString() }} kg</strong> of {{ data.crop }}
        </div>

        <!-- Profit Box -->
        <div class="bg-white/60 border border-white/80 rounded-2xl p-6 mb-5 flex items-center justify-between flex-wrap gap-4 shadow-sm">
          <div>
            <div class="text-xs font-bold text-slate-500 uppercase tracking-widest">Your Net Profit</div>
            <div class="text-xs text-slate-400 mt-1">After transport & all costs</div>
          </div>
          <div class="text-right">
            <div class="text-4xl md:text-5xl font-black text-green-700 leading-none drop-shadow-sm">
              {{ formatCurrency(best.net_profit) }}
            </div>
          </div>
        </div>

        <!-- Stats Grid -->
        <div class="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <div class="bg-white/50 border border-white/60 rounded-xl p-3 text-center shadow-sm">
            <div class="text-lg font-extrabold text-slate-800">{{ formatCurrency(best.current_price) }}/kg</div>
            <div class="text-[10px] text-slate-500 mt-1 uppercase tracking-wider font-bold">Current Price</div>
          </div>
          <div class="bg-white/50 border border-white/60 rounded-xl p-3 text-center shadow-sm">
            <div class="text-lg font-extrabold text-slate-800">{{ formatCurrency(best.gross_revenue) }}</div>
            <div class="text-[10px] text-slate-500 mt-1 uppercase tracking-wider font-bold">Gross Rev</div>
          </div>
          <div class="bg-white/50 border border-white/60 rounded-xl p-3 text-center shadow-sm">
            <div class="text-lg font-extrabold text-slate-800">{{ formatCurrency(best.transport_cost) }}</div>
            <div class="text-[10px] text-slate-500 mt-1 uppercase tracking-wider font-bold">Transport</div>
          </div>
          <div v-if="best.predicted_price_7d" class="bg-white/50 border border-white/60 rounded-xl p-3 text-center shadow-sm">
            <div class="text-lg font-extrabold text-slate-800">{{ formatCurrency(best.predicted_price_7d) }}/kg</div>
            <div class="text-[10px] text-slate-500 mt-1 uppercase tracking-wider font-bold">7-Day Forecast</div>
          </div>
          <div v-else class="bg-white/50 border border-white/60 rounded-xl p-3 text-center shadow-sm">
            <div class="text-lg font-extrabold text-slate-800">{{ formatCurrency(best.storage_cost) }}</div>
            <div class="text-[10px] text-slate-500 mt-1 uppercase tracking-wider font-bold">Storage</div>
          </div>
        </div>

        <div class="mt-5 bg-white/70 border border-amber-300 rounded-xl py-3 px-4 text-sm font-bold text-amber-900 flex justify-center gap-2 shadow-sm">
          <span>📌</span> {{ best.recommendation }}
        </div>

      </div>
    </div>

    <!-- ALTERNATIVES -->
    <template v-if="alternatives.length > 0">
      <h4 class="text-xs font-bold uppercase tracking-widest text-slate-500 mb-3">📋 Other Options (lower profit)</h4>
      <div class="flex flex-col gap-3">
        <div 
          v-for="(alt, i) in alternatives" :key="alt.mandi_name"
          class="bg-white border border-[#dcd1c1] rounded-xl p-4 flex items-center gap-4 animate-[fadeUp_0.4s_ease] transition-all hover:bg-[#f5f0e6] shadow-sm hover:shadow-md"
        >
          <div class="w-8 h-8 bg-[#e8e2d8] rounded-full flex justify-center items-center font-extrabold text-sm text-slate-700 flex-shrink-0 border border-[#dcd1c1]">
            {{ i + 2 }}
          </div>
          <div class="flex-1">
            <div class="font-bold text-sm text-slate-900">{{ alt.mandi_name }}</div>
            <div class="text-xs text-slate-500 mt-1">{{ formatCurrency(alt.current_price) }}/kg &nbsp;·&nbsp; {{ alt.recommendation }}</div>
          </div>
          <div class="text-right">
            <div class="text-base font-extrabold text-green-700">{{ formatCurrency(alt.net_profit) }}</div>
            <div class="text-[11px] text-slate-500 mt-1">{{ alt.distance_km.toFixed(0) }} km away</div>
          </div>
        </div>
      </div>
    </template>

  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  data: any
}>()

const best = computed(() => props.data.best_option)
const alternatives = computed(() => props.data.alternatives)

function formatCurrency(val: number) {
  return '₹' + Math.floor(val).toLocaleString('en-IN')
}
</script>

<style>
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes borderPulse {
  0%, 100% { opacity: 0.7; filter: blur(0px); }
  50% { opacity: 1; filter: blur(3px); }
}
</style>
