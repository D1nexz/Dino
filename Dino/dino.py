"""Dino NEAT — обучение динозаврика в реальном времени.

Одна файл, всё в одном. NEAT обучает сеть прыгать через препятствия
в браузере (Яндекс / Chromium через Playwright).
Запуск:  python dino.py          # обучение в N браузеров, смотреть в окнах
         python dino.py --play    # показать лучшего в одном окне
"""

import argparse
import asyncio
import logging
import os
import pickle
import signal

import neat
from playwright.async_api import async_playwright

logger = logging.getLogger("dino")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ── настройка ──────────────────────────────────────────────────────────
GAME_URL = "https://wayou.github.io/t-rex-runner/"
YANDEX_PATH = "/Applications/Yandex.app/Contents/MacOS/Yandex"

NUM_BROWSERS = 4          # сколько окон браузера открыть параллельно
MAX_GENERATIONS = 200     # сколько поколений обучать
MAX_STEPS = 2000          # шагов на одного динозавра за поколение
STEP_DELAY = 0.0          # задержка между шагами (0 = реальное время игры)
HEADLESS = False          # True — без окон, False — смотреть как динозавр бежит
SAVE_INTERVAL = 5         # каждые N поколений сохранять чекпоинт
BEST_PATH = "best_genome.pkl"
CHECKPOINT_DIR = "checkpoints"

NEAT_INI = """[NEAT]
fitness_criterion     = max
fitness_threshold     = 100000
pop_size              = 10
reset_on_extinction   = False
no_fitness_termination = False

[DefaultGenome]
activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh

aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

bias_default            = 0.0
bias_max_value          = 1.0
bias_min_value          = -1.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_init_type          = gaussian

genome_type             = default_genome

response_default        = 1.0
response_max_value      = 1.0
response_min_value      = -1.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_init_type      = gaussian

conn_add_prob           = 0.5
conn_delete_prob        = 0.5
enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

num_hidden              = 0
num_inputs              = 4
num_outputs             = 1

weight_default          = 0.0
weight_max_value        = 1.0
weight_min_value        = -1.0
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_init_type        = gaussian

node_add_prob           = 0.2
node_delete_prob        = 0.2
single_structural_mutation = False
structural_mutation_surer   = default
initial_connection        = unconnected

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""


class Game:
    """Один браузер + одна игра T-Rex Runner."""

    def __init__(self, browser_id: int, browser, page, context=None):
        self.browser_id = browser_id
        self.browser = browser
        self.page = page
        self.context = context
        self.busy = False

    async def start(self):
        await self.page.goto(GAME_URL, wait_until="domcontentloaded", timeout=60000)
        await self.page.wait_for_timeout(2000)

    async def get_state(self):
        try:
            r = await self.page.evaluate(
                """
                () => {
                    const runner = window.Runner?.instance_;
                    if (!runner) return {status: "NO_RUNNER"};
                    if (runner.crashed) return {status: "DEAD"};
                    if (!runner.playing) return {status: "NOT_PLAYING"};
                    const dino = runner.tRex;
                    const obs = runner.horizon?.obstacles || [];
                    if (obs.length === 0) {
                        return {status: "OK", distance: 1000, type: 0, y: 0, speed: runner.currentSpeed || 1, score: runner.distanceRan || 0};
                    }
                    const o = obs[0];
                    let t = 0;
                    if (o.type && o.type.includes('bird')) t = 2;
                    else if (o.type && o.type.includes('cactus')) t = 1;
                    return {status: "OK", distance: o.xPos - dino.xPos, type: t, y: dino.yPos || 0,
                            speed: runner.currentSpeed || 1, score: runner.distanceRan || 0};
                }
                """
            )
            return r
        except Exception:
            return {"status": "ERROR"}

    async def start_game(self):
        # Запускаем игру — пробел.
        # #main-frame-error НЕ скрываем: это контейнер (.interstitial-wrapper),
        # внутри которого живёт канвас. Его display:none обнуляет размеры
        # канваса (0x0) — звуки идут, а динозаврика не видно.
        await self.page.keyboard.press("Space")
        await self.page.wait_for_timeout(800)
        # Страховка: если CSS-интро не завершилось (не пришло
        # webkitAnimationEnd), выставляем игру в стартовое состояние вручную.
        try:
            await self.page.evaluate(
                """
                () => {
                    const r = window.Runner?.instance_;
                    if (r && r.playingIntro) { r.tRex.playingIntro = false; r.startGame(); }
                }
                """
            )
        except Exception:
            pass
        await self.page.wait_for_timeout(700)
        state = await self.get_state()
        # Если предыдущий забег только что закончился, игра принимает Space не
        # раньше GAMEOVER_CLEAR_TIME (750 мс) — ждём и пробуем старт ещё раз.
        if state.get("status") == "DEAD":
            await self.page.wait_for_timeout(900)
            await self.page.keyboard.press("Space")
            await self.page.wait_for_timeout(500)
            state = await self.get_state()
        # Диагностика: канвас должен иметь ненулевой размер, иначе не видно игры.
        box = None
        try:
            box = await self.page.evaluate(
                """
                () => {
                    const c = document.querySelector('.runner-canvas');
                    if (!c) return null;
                    const b = c.getBoundingClientRect();
                    return [b.x, b.y, b.width, b.height];
                }
                """
            )
        except Exception:
            pass
        if not box or box[2] == 0:
            logger.warning("Браузер %d: канвас не виден (rect=%s)", self.browser_id, box)
        return state

    async def jump(self):
        await self.page.keyboard.press("Space")

    async def close(self):
        try:
            await self.context.close()
        except Exception:
            pass
        try:
            await self.browser.close()
        except Exception:
            pass


def inputs_of(state):
    dist = min(1.0, max(0.0, float(state.get("distance", 1000)) / 1000.0))
    dino_y = 1.0 if float(state.get("y", 0)) > 0 else 0.0
    speed = min(1.0, max(0.0, float(state.get("speed", 1)) / 12.0))
    otype = float(state.get("type", 0)) / 3.0
    return [dist, dino_y, speed, otype]


class Trainer:
    def __init__(self, num_browsers, max_generations, max_steps, step_delay, headless):
        self.num_browsers = num_browsers
        self.max_generations = max_generations
        self.max_steps = max_steps
        self.step_delay = step_delay
        self.headless = headless
        self.neat_config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
            neat.DefaultStagnation, "neat_config.ini",
        )
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        self.gen = 0
        self.running = True
        self.games = []
        self.playwright = None
        self.best_fit = 0.0
        self.best_genome = None

    async def start_browsers(self):
        self.playwright = await async_playwright().start()
        for i in range(self.num_browsers):
            browser = await self.playwright.chromium.launch(
                headless=self.headless, executable_path=YANDEX_PATH,
                args=["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"],
            )
            context = await browser.new_context(viewport={"width": 800, "height": 600})
            page = await context.new_page()
            game = Game(i, browser, page, context)
            await game.start()
            self.games.append(game)
            logger.info("Браузер %d открыт", i)

    async def stop_browsers(self):
        for g in self.games:
            await g.close()
        if self.playwright:
            await self.playwright.stop()
        self.games = []

    def make_agent(self, genome, gid):
        net = neat.nn.FeedForwardNetwork.create(genome, self.neat_config)
        return {"genome": genome, "net": net, "fitness": 0.0, "alive": True,
                "steps": 0, "score": 0, "jumps": 0, "gid": gid}

    async def eval_one(self, game, agent, max_steps):
        state = await game.start_game()
        if state.get("status") != "OK":
            logger.warning("Браузер %d: игра не запустилась (%s)", game.browser_id, state.get("status"))
            await asyncio.sleep(2.0)
            state = await game.start_game()
            if state.get("status") != "OK":
                return 0.0

        agent["alive"] = True
        agent["fitness"] = 0.0
        agent["steps"] = 0
        agent["score"] = 0
        agent["jumps"] = 0

        for _ in range(max_steps):
            if not agent["alive"]:
                break
            state = await game.get_state()
            status = state.get("status")
            if status == "ERROR":
                # Страница закрылась — пересоздаём и перезапускаем игру
                logger.warning("Браузер %d: страница закрыта, пересоздаём...", game.browser_id)
                try:
                    await game.page.close()
                except Exception:
                    pass
                try:
                    game.context = await game.browser.new_context(viewport={"width": 800, "height": 600})
                    game.page = await game.context.new_page()
                    await game.page.goto(GAME_URL, wait_until="domcontentloaded", timeout=60000)
                    await game.page.wait_for_timeout(2000)
                except Exception as exc:
                    logger.error("Не удалось пересоздать страницу: %s", exc)
                    return 0.0
                state = await game.start_game()
                if state.get("status") != "OK":
                    return 0.0
                continue
            if status == "DEAD":
                agent["alive"] = False
                agent["fitness"] -= 50.0
                break
            if status not in ("OK", "NOT_PLAYING"):
                await asyncio.sleep(self.step_delay)
                continue

            do_jump = agent["net"].activate(inputs_of(state))[0] > 0.5
            if do_jump:
                await game.jump()
                agent["jumps"] += 1

            agent["steps"] += 1
            agent["score"] = int(state.get("score", 0))
            agent["fitness"] += 0.1

            d = float(state.get("distance", 1000))
            if d < 40 and do_jump:
                agent["fitness"] += 2.0
            elif d < 40 and not do_jump:
                agent["fitness"] -= 2.0

            await asyncio.sleep(self.step_delay)

        fit = agent["fitness"] + agent["steps"] * 0.05 + agent["score"] * 0.5
        agent["genome"].fitness = fit
        return fit

    async def eval_generation(self, genomes):
        """Оценить всех геномов текущего поколения параллельно в браузерах."""
        agents = {gid: self.make_agent(g, gid) for gid, g in genomes}
        sem = asyncio.Semaphore(self.num_browsers)

        async def run_one(gid, genome):
            async with sem:
                game = None
                for g in self.games:
                    if not g.busy:
                        g.busy = True
                        game = g
                        break
                if game is None:
                    genome.fitness = 0.0
                    return
                try:
                    await self.eval_one(game, agents[gid], self.max_steps)
                finally:
                    game.busy = False

        await asyncio.gather(*(run_one(gid, g) for gid, g in genomes))

        fits = [a["fitness"] for a in agents.values() if a["fitness"] > 0]
        if fits:
            best_fit = max(fits)
            best_agent = max(agents.values(), key=lambda a: a["fitness"])
            if best_fit > self.best_fit:
                self.best_fit = best_fit
                self.best_genome = best_agent["genome"]
                with open(BEST_PATH, "wb") as f:
                    pickle.dump(self.best_genome, f)
                logger.info("Поколение %d: лучший фитнесс %.2f (геном %d)", self.gen, best_fit, best_agent["gid"])
        self.gen += 1

        # Следующее поколение NEAT (селекция + мутация)
        new_pop = self.pop.reproduction.reproduce(
            self.neat_config, self.pop.species, self.neat_config.pop_size, self.gen
        )
        if new_pop:
            self.pop.population = new_pop
            self.pop.species.speciate(self.neat_config, self.pop.population, self.gen)
        else:
            self.pop.population = self.pop.reproduction.create_new(
                self.neat_config.genome_type, self.neat_config.genome_config, self.neat_config.pop_size
            )
            self.pop.species.speciate(self.neat_config, self.pop.population, self.gen)

        if self.gen % SAVE_INTERVAL == 0:
            self.save_checkpoint()

        if self.best_fit >= self.neat_config.fitness_threshold:
            self.running = False

    def save_checkpoint(self):
        path = os.path.join(CHECKPOINT_DIR, f"neat_checkpoint_gen_{self.gen}.pkl")
        data = {"generation": self.gen, "best_fit": getattr(self, "best_fit", 0.0),
                "population": self.pop.population if self.pop else {}}
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Чекпоинт: %s", path)

    def load_checkpoint(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.gen = data.get("generation", 0)
        self.best_fit = data.get("best_fit", 0.0)
        self.pop.population = data.get("population", {})
        # Пересоздаём виды для загруженной популяции
        self.pop.species.speciate(self.neat_config, self.pop.population, self.gen)

    async def train(self):
        await self.start_browsers()
        try:
            self.pop = neat.Population(self.neat_config)

            cp = os.path.join(CHECKPOINT_DIR, f"neat_checkpoint_gen_{self.gen}.pkl")
            if os.path.exists(cp):
                self.load_checkpoint(cp)
                logger.info("Загружен чекпоинт %s", cp)
            else:
                logger.info("Новая популяция NEAT")

            while self.running and self.gen < self.max_generations:
                logger.info("=== Поколение %d ===", self.gen)
                await self.eval_generation(list(self.pop.population.items()))
                if not self.running:
                    break
        finally:
            await self.stop_browsers()
        logger.info("Обучение завершено. Лучший фитнесс: %.2f", getattr(self, "best_fit", 0.0))

    def stop(self):
        self.running = False


async def play_best():
    """Показать лучшего динозавра в одном окне (реальное время)."""
    if not os.path.exists(BEST_PATH):
        logger.error("best_genome.pkl не найден. Сначала запустите обучение.")
        return
    with open(BEST_PATH, "rb") as f:
        genome = pickle.load(f)

    neat_config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
        neat.DefaultStagnation, "neat_config.ini",
    )
    net = neat.nn.FeedForwardNetwork.create(genome, neat_config)

    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(
        headless=False, executable_path=YANDEX_PATH,
        args=["--disable-gpu", "--no-sandbox"],
    )
    context = await browser.new_context(viewport={"width": 800, "height": 600})
    page = await context.new_page()
    game = Game(0, browser, page)
    game.context = context
    await game.start()

    state = await game.start_game()
    logger.info("Игра запущена, статус: %s", state.get("status"))

    steps = 0
    alive = True
    while alive and steps < 5000:
        state = await game.get_state()
        status = state.get("status")
        if status == "DEAD":
            logger.info("Смерть после %d шагов, счёт: %d", steps, int(state.get("score", 0)))
            break
        if status == "ERROR":
            logger.warning("Страница закрыта, остановка просмотра.")
            break
        if status not in ("OK", "NOT_PLAYING"):
            await asyncio.sleep(0.03)
            continue

        do_jump = net.activate(inputs_of(state))[0] > 0.5
        if do_jump:
            await game.jump()

        steps += 1
        await asyncio.sleep(0.03)

    logger.info("Итог: шагов=%d", steps)
    await game.close()
    await playwright.stop()


def main():
    parser = argparse.ArgumentParser(description="Dino NEAT — обучение в реальном времени")
    parser.add_argument("--play", action="store_true", help="показать лучшего в одном окне")
    parser.add_argument("--browsers", type=int, default=NUM_BROWSERS)
    parser.add_argument("--generations", type=int, default=MAX_GENERATIONS)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--step-delay", type=float, default=STEP_DELAY)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if args.play:
        asyncio.run(play_best())
        return

    trainer = Trainer(args.browsers, args.generations, args.max_steps, args.step_delay, args.headless)

    def on_sig():
        trainer.stop()
        stop_event.set()
        logger.info("Остановка по сигналу...")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, on_sig)
            except (NotImplementedError, RuntimeError):
                pass

    asyncio.run(trainer.train())


if __name__ == "__main__":
    main()
