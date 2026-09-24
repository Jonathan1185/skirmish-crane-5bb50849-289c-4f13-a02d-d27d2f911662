"""A working Crane starter agent.

Each unit runs a separate instance of this class. This starter walks forward until it sees an
enemy, then takes one legal step toward the nearest visible enemy and names it. Start at the
``TODO(you)`` comments.
Read ``environment.md`` beside this file for the rules, helpers, and first improvement. Prepare
episode state in ``reset``. The constructor takes no arguments.
"""

from heapq import heappop, heappush

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
        # A fixed destination on the opposing side gives melee units a goal before
        # an enemy enters vision, including when terrain blocks the direct route.
        self.advance_goal = tile.at_mirror(me.position(observation), observation)

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
            if me.unit_type(observation) in {"footman", "cavalry"}:
                advance = self._path_toward(observation, self.advance_goal)
                if advance:
                    return action.move(advance)

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

        # Footmen and cavalry use their full legal route toward the nearest visible enemy.

        path = self._path_toward(observation, nearest["position"])

        # Naming a target makes the strike prefer that enemy. Any visible enemy can be named,
        # so both orders below are legal.
        if path == 0:
            return action.stay(nearest["unit_id"], observation)
        return action.move(path, nearest["unit_id"], observation)

    def _path_toward(self, observation: SkirmishObservation, goal: AxialPosition) -> int:
        """Return the legal path with the shortest terrain route remaining to goal."""
        here = me.position(observation)
        routes = [path_id for path_id in action.legal_paths(observation) if path_id != 0]
        if not routes:
            return 0

        route_distances = self._terrain_distances(observation, goal)

        # Trying every affordable path lets a unit spend its movement points. The
        # terrain route distance, rather than straight-line distance, sends units
        # around water and other impassable map features.
        return min(
            routes,
            key=lambda path_id: (
                route_distances.get(
                    self._position_key(tile.at_path_end(here, path_id)), float("inf")
                ),
                -len(paths.decode(path_id)),
            ),
        )

    def _terrain_distances(
        self, observation: SkirmishObservation, goal: AxialPosition
    ) -> dict[tuple[int, int], int]:
        """Map every passable tile to its terrain-aware movement cost to goal."""
        goal_key = self._position_key(goal)
        distances = {goal_key: 0}
        queue = [(0, goal_key)]

        # Work backwards from the goal. This means moving from a tile to its neighbor
        # pays the cost to enter that neighbor, matching the game's movement rules.
        while queue:
            distance, current_key = heappop(queue)
            if distance != distances[current_key]:
                continue

            current = {"q": current_key[0], "r": current_key[1]}
            entry_cost = self._movement_cost(observation, current)
            for neighbor in tile.neighbors(current).values():
                neighbor_key = self._position_key(neighbor)
                if self._movement_cost(observation, neighbor) is None:
                    continue

                candidate = distance + entry_cost
                if candidate < distances.get(neighbor_key, float("inf")):
                    distances[neighbor_key] = candidate
                    heappush(queue, (candidate, neighbor_key))

        return distances

    @staticmethod
    def _position_key(position: AxialPosition) -> tuple[int, int]:
        return position["q"], position["r"]

    @staticmethod
    def _movement_cost(observation: SkirmishObservation, position: AxialPosition) -> int | None:
        """Return the movement cost to enter position, or None for impassable tiles."""
        terrain = tile.terrain_at(observation, position)
        if terrain["terrain"] in {"void", "water"}:
            return None

        cost = 2 if terrain["terrain"] == "hill" else 1
        if terrain["feature"] == "forest":
            cost += 1
        elif terrain["feature"] == "marsh":
            cost += 2
        return cost

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
