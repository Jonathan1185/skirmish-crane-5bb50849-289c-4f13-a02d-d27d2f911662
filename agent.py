"""A working Crane starter agent.

Each unit runs a separate instance of this class. This starter walks forward until it sees an
enemy, then takes one legal step toward the nearest visible enemy and names it. Start at the
``TODO(you)`` comments.
Read ``environment.md`` beside this file for the rules, helpers, and first improvement. Prepare
episode state in ``reset``. The constructor takes no arguments.
"""

from sandbox.crane import action, me, paths, tile, units, visible
from sandbox.observation_types import AxialPosition, SkirmishAction, SkirmishObservation


class Agent:
    """Marches toward the enemy side, then steps toward the nearest visible enemy."""

    def reset(self, seed, observation) -> None:
        # Remember ally locations so an archer can fall back toward an ally even after
        # that ally has moved outside the archer's current vision.
        self.ally_positions = {
            ally["unit_id"]: ally["position"]
            for ally in visible.allies(observation)
        }

    def act(self, observation: SkirmishObservation) -> SkirmishAction:
        # Refresh any allies currently in view, while retaining their last known locations.
        self.ally_positions.update(
            {
                ally["unit_id"]: ally["position"]
                for ally in visible.allies(observation)
            }
        )

        # The enemies this unit can see.
        enemies = visible.enemies(observation)

        if not enemies:
            # At the beginning of a default skirmish match, units sit apart and see no enemies.
            # me.direction is the digit toward the enemy side, so this unit heads that way.
            forward = me.direction(observation)

            # legal_steps lists the single steps allowed by the mask. Checking membership keeps
            # this order legal when a wall, ally, or enemy blocks the way.
            if forward in action.legal_steps(observation):
                return action.move(forward)

            # TODO(you): this unit stands still when something blocks the way.
            # It may still attack, but can you choose a better response?
            return action.stay()

        # This unit's current {"q": ..., "r": ...} position.
        here = me.position(observation)

        # The closest enemy in sight. min returns the enemy dictionary, not the distance.
        nearest = min(enemies, key=lambda enemy: tile.distance(here, enemy["position"]))

        # Archers are fragile. Fall back toward the nearest ally's last known location
        # only when the closest enemy is already inside bow range.
        if me.unit_type(observation) == "archer":
            enemy_distance = tile.distance(here, nearest["position"])
            archer_range = units.STATS["archer"].attack_range
            if enemy_distance < archer_range:
                if self.ally_positions:
                    nearest_ally_position = min(
                        self.ally_positions.values(),
                        key=lambda position: tile.distance(here, position),
                    )
                else:
                    nearest_ally_position = None

                # Escape from every visible enemy, maximizing the distance gained.
                # Allied support breaks ties between equally safe tiles.
                retreat = self._step_away_from_enemies(
                    observation, enemies, nearest_ally_position
                )
                if retreat:
                    return action.move(retreat, nearest["unit_id"], observation)
            return action.stay(nearest["unit_id"], observation)

        # Footmen and cavalry continue to advance on the nearest visible enemy.

        # The step that gets closest to the enemy, or 0 when no step gets closer.
        step = self._step_toward(observation, nearest["position"])

        # Naming a target makes the strike prefer that enemy. Any visible enemy can be named,
        # so both orders below are legal.
        if step == 0:
            return action.stay(nearest["unit_id"], observation)
        return action.move(step, nearest["unit_id"], observation)

    def _step_toward(self, observation: SkirmishObservation, goal: AxialPosition) -> int:
        """Return the single step that most closes the gap to goal, or 0 when none does."""
        # TODO(you): only single steps are tried here. A path can contain four steps, and cavalry
        # has four movement points, so most of that speed goes to waste.
        here = me.position(observation)

        # Standing still is path id 0. A step must reduce the distance to be worth taking.
        best_step = 0
        best_distance = tile.distance(here, goal)

        for step in action.legal_steps(observation):
            # at_path_end gives the landing tile, so this is the distance after the step.
            step_distance = tile.distance(tile.at_path_end(here, step), goal)

            # Remember this step if it is the best one so far.
            if step_distance < best_distance:
                best_step, best_distance = step, step_distance

        return best_step

    def _step_away_from_enemies(
        self,
        observation: SkirmishObservation,
        enemies: list[dict],
        ally_position: AxialPosition | None,
    ) -> int:
        """Return a legal path that maximizes distance from visible enemies."""
        here = me.position(observation)
        current_enemy_distance = min(
            tile.distance(here, enemy["position"]) for enemy in enemies
        )
        escape_paths = []

        # legal_paths includes every route this archer can afford this turn, not
        # only one-tile moves. Path 0 is standing still, so it cannot be an escape.
        for path_id in action.legal_paths(observation):
            if path_id == 0:
                continue
            landing = tile.at_path_end(here, path_id)
            enemy_distance = min(
                tile.distance(landing, enemy["position"]) for enemy in enemies
            )
            if enemy_distance > current_enemy_distance:
                ally_distance = (
                    tile.distance(landing, ally_position)
                    if ally_position is not None
                    else 0
                )
                escape_paths.append(
                    (path_id, enemy_distance, len(paths.decode(path_id)), ally_distance)
                )

        if not escape_paths:
            return 0

        # Maximize distance from enemies, then use as many legal steps as possible.
        # Only use ally distance to break a remaining tie.
        return max(
            escape_paths, key=lambda candidate: (candidate[1], candidate[2], -candidate[3])
        )[0]

    # Optional: a reinforcement-learning hook called after every step with that step's
    # transition. Its time counts against the timing and episode budget. The order argument is
    # what act returned. It is named order so it does not shadow the action helpers.
    #
    # def learn(self, observation, order: SkirmishAction, reward: float, terminated: bool) -> None:
    #     ...

    # Optional: messaging. Season settings enable it from Season 3 onward. When enabled, chat runs
    # after a unit chooses its order and receives messages that arrived since its previous
    # activation. Return each message with a recipient and text. Use None to broadcast to both
    # sides, or a player id such as "player_2", not a unit id, to send directly to one ally. The
    # rosters in the observation map each player to its unit. By default, text is limited to 200
    # characters.
    # A direct message reaches its allied unit at its next activation, after that unit chooses its
    # own order. Every message is recorded and shown in replays, so nothing you send is ever secret.
    # Return nothing to stay silent.
    #
    # def chat(self, inbox: list[dict]) -> list[dict] | None:
    #     ...
