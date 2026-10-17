<template>
  <div class="w-full max-w-[680px] mt-8 mb-16">

    <!-- Summary Banner -->
    <div class="bg-amber-50 border border-amber-200 rounded-2xl p-5 text-sm text-amber-900 mb-8 leading-relaxed flex items-start gap-3 animate-[fadeUp_0.3s_ease] shadow-sm">
      <span class="text-xl mt-0.5">💡</span>
      <span>{{ data.summary }}</span>
    </div>

    <!-- ═══════════════════════════════════════════════ -->
    <!--       BEST RECOMMENDATION — HERO CARD          -->
    <!-- ═══════════════════════════════════════════════ -->
    <div class="relative mb-10 animate-[fadeUp_0.35s_ease] group">
      <!-- Animated glowing border ring -->
      <div class="absolute -inset-[2px] rounded-[1.6rem] bg-gradient-to-br from-green-500 via-emerald-400 to-green-600 z-0 opacity-60 blur-0 animate-[borderPulse_3s_ease-in-out_infinite]"></div>

      <div class="relative z-10 bg-gradient-to-br from-[#faf7ef] to-[#f0e5cd] rounded-[1.5rem] p-8 overflow-hidden border border-[#d9c59a]/60 shadow-xl">
        <!-- Soft inner glow -->
        <div class="absolute -top-12 -right-12 w-56 h-56 rounded-full bg-green-200/20 blur-3xl pointer-events-none"></div>
        <div class="absolute -bottom-8 -left-8 w-40 h-40 rounded-full bg-amber-200/20 blur-3xl pointer-events-none"></div>

        <!-- Badge -->
        <div class="inline-flex items-center gap-2 bg-gradient-to-r from-green-600 to-emerald-500 text-white text-[10px] font-black tracking-[0.2em] uppercase rounded-full px-4 py-1.5 mb-5 shadow-md shadow-green-600/25">
          🏆 #1 Best Mandi
        </div>

        <!-- Mandi Name -->
        <h3 class="text-4xl md:text-5xl font-black leading-none text-slate-900 drop-shadow-sm mb-1.5">
          {{ best.mandi_name }}
        </h3>

        <div class="text-sm text-[#8b7d6b] mb-7 flex flex-wrap items-center gap-x-2 gap-y-1">
          <span class="inline-flex items-center gap-1">📍 {{ best.distance_km.toFixed(0) }} km away</span>
          <span class="text-[#ccc]">·</span>
          <span>Sell <strong class="text-slate-700">{{ data.quantity.toLocaleString() }} kg</strong> of {{ data.crop }}</span>
        </div>

        <!-- ─── NET PROFIT HERO BOX ─── -->
        <div class="bg-white/70 backdrop-blur-sm border border-white/80 rounded-2xl p-6 mb-6 flex items-center justify-between flex-wrap gap-4 shadow-sm">
          <div>
            <div class="text-[10px] font-black text-slate-500 uppercase tracking-[0.15em]">Your Net Profit</div>
            <div class="text-xs text-[#8b7d6b] mt-0.5">After all transport, storage & spoilage costs</div>
          </div>
          <div class="text-right">
            <div class="text-4xl md:text-5xl font-black text-green-700 leading-none drop-shadow-sm tracking-tight">
              {{ formatCurrency(best.net_profit) }}
            </div>
          </div>
        </div>

        <!-- ─── COST BREAKDOWN GRID ─── -->
        <div class="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-5">
          <div class="bg-white/50 border border-white/60 rounded-xl p-3.5 text-center shadow-sm">
            <div class="text-lg font-extrabold text-slate-800">{{ formatCurrency(best.current_price) }}/kg</div>
            <div class="text-[10px] text-slate-500 mt-1 uppercase tracking-wider font-bold">Market Price</div>
          </div>
          <div class="bg-white/50 border border-white/60 rounded-xl p-3.5 text-center shadow-sm">
            <div class="text-lg font-extrabold text-slate-800">{{ formatCurrency(best.gross_revenue) }}</div>
            <div class="text-[10px] text-slate-500 mt-1 uppercase tracking-wider font-bold">Gross Revenue</div>
          </div>
          <div class="bg-white/50 border border-white/60 rounded-xl p-3.5 text-center shadow-sm">
            <div class="text-lg font-extrabold text-red-600">-{{ formatCurrency(best.transport_cost) }}</div>
            <div class="text-[10px] text-slate-500 mt-1 uppercase tracking-wider font-bold">Transport</div>
          </div>
          <div v-if="best.predicted_price_7d" class="bg-white/50 border border-white/60 rounded-xl p-3.5 text-center shadow-sm">
            <div class="text-lg font-extrabold text-slate-800">{{ formatCurrency(best.predicted_price_7d) }}/kg</div>
            <div class="text-[10px] text-slate-500 mt-1 uppercase tracking-wider font-bold">7d Forecast</div>
          </div>
          <div v-else class="bg-white/50 border border-white/60 rounded-xl p-3.5 text-center shadow-sm">
            <div class="text-lg font-extrabold text-red-600">-{{ formatCurrency(best.storage_cost) }}</div>
            <div class="text-[10px] text-slate-500 mt-1 uppercase tracking-wider font-bold">Storage</div>
          </div>
        </div>

        <!-- Action Recommendation -->
        <div class="bg-white/80 border border-amber-200 rounded-xl py-3.5 px-5 text-sm font-bold text-amber-900 flex items-center justify-center gap-2 shadow-sm">
          <span>📌</span> {{ best.recommendation }}
        </div>

      </div>
    </div>

    <!-- ═══════════════════════════════════════════════ -->
    <!--            ALTERNATIVE OPTIONS                 -->
    <!-- ═══════════════════════════════════════════════ -->
    <template v-if="alternatives.length > 0">
      <h4 class="text-xs font-black uppercase tracking-[0.15em] text-[#8b7d6b] mb-4 flex items-center gap-2">
        <span class="w-5 h-[2px] bg-[#ccc] rounded-full inline-block"></span>
        Other Options (lower profit)
      </h4>
      <div class="flex flex-col gap-3">
        <div
          v-for="(alt, i) in alternatives" :key="alt.mandi_name"
          class="bg-white/70 backdrop-blur-sm border border-[#dcd1c1] rounded-xl p-4 flex items-center gap-4 animate-[fadeUp_0.4s_ease] transition-all duration-200 hover:bg-white hover:shadow-lg hover:shadow-green-600/5 hover:border-green-400/30 group/alt"
        >
          <div class="w-9 h-9 bg-[#f0ebe0] rounded-full flex justify-center items-center font-extrabold text-sm text-slate-600 flex-shrink-0 border border-[#dcd1c1] group-hover/alt:bg-green-50 group-hover/alt:border-green-200 group-hover/alt:text-green-700 transition-colors duration-200">
            {{ Number(i) + 2 }}
          </div>
          <div class="flex-1 min-w-0">
            <div class="font-bold text-sm text-slate-900">{{ alt.mandi_name }}</div>
            <div class="text-xs text-[#8b7d6b] mt-0.5 truncate">{{ formatCurrency(alt.current_price) }}/kg · {{ alt.recommendation }}</div>
          </div>
          <div class="text-right flex-shrink-0">
            <div class="text-base font-extrabold text-green-700">{{ formatCurrency(alt.net_profit) }}</div>
            <div class="text-[11px] text-[#8b7d6b] mt-0.5">{{ alt.distance_km.toFixed(0) }} km</div>
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
  0%, 100% { opacity: 0.55; filter: blur(0px); }
  50% { opacity: 0.9; filter: blur(4px); }
}
</style>
