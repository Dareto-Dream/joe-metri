/* Shared Geometry Dash AI orchestrator screensaver helpers. */
(function () {
  const POLL_MS = 2500;

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function number(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function formatInt(value) {
    return Math.round(number(value)).toLocaleString();
  }

  function formatFloat(value, digits = 3) {
    const parsed = number(value);
    return parsed > 0 ? parsed.toFixed(digits) : "-";
  }

  function formatPct(value, digits = 0) {
    return `${clamp(number(value), 0, 100).toFixed(digits)}%`;
  }

  function formatRate(value) {
    const parsed = number(value);
    if (parsed <= 0) return "-";
    if (parsed >= 1000000) return `${(parsed / 1000000).toFixed(2)}m`;
    if (parsed >= 1000) return `${(parsed / 1000).toFixed(1)}k`;
    return `${Math.round(parsed)}`;
  }

  function pad(value) {
    return String(value).padStart(2, "0");
  }

  function formatDuration(totalSeconds) {
    const parsed = Math.max(0, Math.floor(number(totalSeconds)));
    const hours = Math.floor(parsed / 3600);
    const minutes = Math.floor((parsed % 3600) / 60);
    const seconds = parsed % 60;
    return `${hours}:${pad(minutes)}:${pad(seconds)}`;
  }

  function latestEvent(events) {
    if (!Array.isArray(events) || events.length === 0) return "WAITING_FOR_ORCHESTRATOR";
    const item = events[events.length - 1] || {};
    return String(item.event || "ORCHESTRATOR_EVENT");
  }

  function normalize(raw = {}) {
    const orchestrator = raw.orchestrator || {};
    const metrics = orchestrator.metrics || {};
    const dashboard = metrics.dashboard || {};
    const queues = dashboard.queue_sizes || {};
    const tokenizer = metrics.tokenizer || {};
    const trainer = metrics.trainer || {};
    const scraper = metrics.scraper || {};
    const allocation = metrics.allocation || {};
    const training = raw.training || {};
    const latestTraining = orchestrator.latest_training || {};
    const latestEvaluation = orchestrator.latest_evaluation || {};

    return {
      connected: true,
      generatedAt: number(raw.generated_at),
      active: Boolean(orchestrator.active || training.active || training.live),
      shutdownAllowed: Boolean(raw.shutdown_allowed),
      mode: String(orchestrator.mode || metrics.mode || "unknown"),
      latestEvent: latestEvent(orchestrator.events),
      events: Array.isArray(orchestrator.events) ? orchestrator.events : [],
      uptimeSeconds: number(metrics.uptime_seconds || training.uptime_seconds),
      system: {
        cpu: clamp(number(raw.cpu), 0, 100),
        ram: clamp(number(raw.ram), 0, 100),
        gpu: clamp(number(raw.gpu), 0, 100),
        vramUsed: number(raw.vram_used),
        vramTotal: number(raw.vram_total),
      },
      dashboard: {
        levelsScraped: number(dashboard.levels_scraped),
        levelsTokenized: number(dashboard.levels_tokenized || training.levels_tokenized),
        tokensGenerated: number(dashboard.tokens_generated || training.tokens_generated),
        tokensPerSecond: number(dashboard.tokens_per_second || training.tokens_per_second),
        datasetSize: number(dashboard.dataset_size || training.dataset_samples),
        validationQuality: number(dashboard.validation_quality),
        trainLoss: number(dashboard.training_loss || training.loss),
        rawQueue: number(queues.raw_queue || training.raw_queue),
        tokenQueue: number(queues.token_queue || training.token_queue),
        trainingQueue: number(queues.training_queue || training.training_queue),
      },
      tokenizer: {
        throughput: number(tokenizer.token_throughput),
        unknownRate: number(tokenizer.unknown_object_rate),
        entropy: number(tokenizer.token_entropy || training.token_entropy),
        stepDensity: number(tokenizer.avg_step_density),
        grammarValidity: number(tokenizer.grammar_validity || training.grammar_validity),
      },
      scraper: {
        levelsPerMinute: number(scraper.levels_per_minute),
        failures: number(scraper.request_failures),
        duplicates: number(scraper.duplicates_skipped),
        sourceDiversity: number(scraper.source_diversity),
      },
      trainer: {
        steps: number(trainer.steps || training.step),
        trainLoss: number(latestTraining.train_loss || trainer.training_loss || training.loss),
        valLoss: number(latestTraining.eval_loss || trainer.validation_loss || training.val_loss),
        diversity: number(latestEvaluation.diversity || trainer.generation_diversity || training.generation_diversity),
        grammarLegal: trainer.grammar_legal,
        collapsed: Boolean(trainer.collapse_indicator || latestEvaluation.collapsed),
      },
      allocation: {
        cpuSlots: number(allocation.cpu_slots),
        gpuSlots: number(allocation.gpu_slots),
        scraperConcurrency: number(allocation.scraper_concurrency),
        tokenizerWorkers: number(allocation.tokenizer_workers),
        tokenizerRecordsPerCycle: number(allocation.tokenizer_records_per_cycle),
        trainingExamplesPerCycle: number(allocation.training_examples_per_cycle),
      },
      samplePreview: String(training.sample_preview || ""),
      vocabSize: number(orchestrator.vocab_size),
      shutdownMessage: String(
        training.shutdown_message ||
        (orchestrator.active
          ? "The orchestrator is running. Use Ctrl+C in the terminal for a graceful shutdown."
          : "No recent orchestrator heartbeat was detected.")
      ),
    };
  }

  function fallback() {
    return normalize({
      shutdown_allowed: true,
      orchestrator: {
        active: false,
        mode: "offline",
        events: [{ event: "WAITING_FOR_ORCHESTRATOR", timestamp: Math.floor(Date.now() / 1000) }],
        metrics: {
          uptime_seconds: 0,
          dashboard: {
            levels_scraped: 0,
            levels_tokenized: 0,
            tokens_generated: 0,
            tokens_per_second: 0,
            queue_sizes: { raw_queue: 0, token_queue: 0, training_queue: 0 },
          },
          tokenizer: { token_entropy: 0, grammar_validity: 0, avg_step_density: 0, unknown_object_rate: 0 },
          trainer: { steps: 0, training_loss: 0, validation_loss: 0, generation_diversity: 0 },
          scraper: { levels_per_minute: 0, request_failures: 0, duplicates_skipped: 0, source_diversity: 0 },
          allocation: {},
        },
      },
      training: {
        active: false,
        live: false,
        shutdown_message: "Start the orchestrator and screensaver server to display live telemetry.",
      },
    });
  }

  function statSources() {
    const local = "http://127.0.0.1:5000/stats";
    if (window.location.protocol === "file:") return [local];
    return Array.from(new Set([new URL("/stats", window.location.href).toString(), local]));
  }

  function pollTelemetry(onPayload) {
    const sources = statSources();
    let sourceIndex = 0;

    async function fetchStats() {
      let lastError = null;
      for (let offset = 0; offset < sources.length; offset += 1) {
        const index = (sourceIndex + offset) % sources.length;
        try {
          const response = await fetch(sources[index], { cache: "no-store" });
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          sourceIndex = index;
          onPayload(normalize(await response.json()));
          return;
        } catch (error) {
          lastError = error;
        }
      }
      if (lastError) console.warn("Unable to fetch orchestrator stats:", lastError);
      const value = fallback();
      value.connected = false;
      onPayload(value);
    }

    fetchStats();
    return window.setInterval(fetchStats, POLL_MS);
  }

  window.GDSaver = {
    clamp,
    fallback,
    formatDuration,
    formatFloat,
    formatInt,
    formatPct,
    formatRate,
    normalize,
    pad,
    pollTelemetry,
  };
})();
