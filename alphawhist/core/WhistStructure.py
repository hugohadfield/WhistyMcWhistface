from alphawhist.core.WhistPlayer import *
import sys
import json
import random
from alphawhist.core.InformationState import InformationState

class Game():
    def __init__(self, playerConfigFileName):
        # Game related variables
        self.cardnumbers = [7,6,5,4,3,2,1,2,3,4,5,6,7]
        self.playerConfigFileName = ""
        self.playerConfigFileName = playerConfigFileName
        self.numberofplayers = 0
        self.players = []
        self.generatePlayersFromConfigFile(self.playerConfigFileName)
        self.resetGame()

    def parseGameConfigFile(self, configFileName):
        with open(configFileName) as data_file:
            jsonFileData = json.load(data_file)
            for parameterCollection in jsonFileData["GameParameters"]:
                playerConfigName = parameterCollection["PlayerConfigFile"]
                self.playerConfigFileName = playerConfigName

    def generatePlayersFromConfigFile(self,configFileName):
        with open(configFileName) as data_file:
            jsonFileData = json.load(data_file)
            for player in jsonFileData["Players"]:
                playerClassName = player["class"]
                playerNumber = player["playerNumber"]

                module = __import__(playerClassName)
                class_ = getattr(module, playerClassName)
                instance = class_(int(playerNumber), additionalParameters=player["additionalParameters"])
                self.players.append(instance)
        self.numberofplayers = len(self.players)

    def generatePlayersFromList(self, playerClassNameList):
        n = 0
        for playerClassName in playerClassNameList:
            module = __import__(playerClassName)
            class_ = getattr(module, playerClassName)
            instance = class_(n)
            self.players.append(instance)
            n = n + 1

    def playFullGame(self):
        for i in range(0, len(self.cardnumbers)):
            print(("Round number: ", i + 1))
            self.playFullRound()
        recorder = getattr(self, "training_recorder", None)
        if recorder is not None:
            recorder.on_game_end(self)

    def playFullRound(self):
        # How many cards are in this round
        cardsinround = self.cardnumbers[self.roundnumber]
        print(" ")
        print(" ")
        print("------------------------------------------------")
        print("Cards in round: ", end=' ')
        print(cardsinround)

        # Reset the game variables
        self.resetRound()
        
        # AI-only games need not ask for manual card input.
        from ManualPlayer import ManualPlayer
        if any(isinstance(player, ManualPlayer) for player in self.players):
            self.manualDeal(cardsinround)
        else:
            self.randomDeal(cardsinround)

        # The dealer picks trumps
        self.trumpsuit = self.players[self.dealer].pick_trumps(self.numberofplayers)
        self._record_public_event({
            "type": "trump", "player": self.dealer, "suit": self.trumpsuit
        })
        print("Trumps: ", self.trumpsuit)
        
        # Get all the bids
        self.getPlayerBids()
        
        # Play all the tricks
        self.turn = self.leader
        for trick in range(0,cardsinround):
            self.playFullTrick()

        self.processScore()
        print((self.scores))
        self.endRound()

    def playPartialRound(self):
         # How many cards are in this round
        cardsinround = self.cardnumbers[self.roundnumber]
        print("Cards in round: ", end=' ')
        print(cardsinround)
        
        # Play the remaining tricks
        startingTurn = self.trickNumber
        for trick in range(startingTurn,cardsinround):
            self.playFullTrick()

        self.processScore()
        print((self.scores))
        self.endRound()


    def playFullTrick(self):
        print(" ")
        print(" ")
        print("########### Beggining full trick number, " , self.trickNumber ,"############")
        print("Bids: ", self.bids)
        print("Player cards ")
        for p in self.players:
            print(p.possible_hand)
        print("#############################################")
        self.pile = []
        for n in range(0,self.numberofplayers):
            player = self.players[ self.turn ]

            # Optional dataset recording: snapshot the actor's information
            # BEFORE its decision. Never pass a Game or hidden hands to the
            # training feature encoder. No observation overhead if disabled.
            recorder = getattr(self, "training_recorder", None)
            snapshot = self.observe(player.playerNumber) if recorder is not None else None
            legal = player.getRealPossibleMoves(self.pile) if recorder is not None else None
            cardPlayed = player.makeMove(self.pile.copy(), self.trumpsuit, self)
            if recorder is not None:
                recorder.on_decision(self, snapshot, legal, cardPlayed)

            self.pile.append( cardPlayed )
            self._record_public_event({
                "type": "card", "player": player.playerNumber, "card": cardPlayed
            })
            print("Player: ", player.playerNumber, " played: ", end=' ')
            print(cardPlayed)

            self.cleanCardFromPlayers(cardPlayed)

            self.incrementTurn()

        # See who has won and move on
        winningplay = self.extractTrickWinner()
        self.updateTricksWon(winningplay)
        self._record_public_event({
            "type": "trick_end",
            "winner": (self.leader + winningplay) % self.numberofplayers
        })
        print("#############################################")
        self.setTurnToWinner()
        self.trickNumber = self.trickNumber + 1
        print(self.tricksWon)
        print("############### End of trick ################")

    def playPartialTrick(self,specificCard):
        player = self.players[ self.turn ]
        recorder = getattr(self, "training_recorder", None)
        snapshot = self.observe(player.playerNumber) if recorder is not None else None
        legal = player.getRealPossibleMoves(self.pile) if recorder is not None else None
        if player.playCard(specificCard):
            if recorder is not None:
                recorder.on_decision(self, snapshot, legal, specificCard)
            self.pile.append(specificCard)
            self._record_public_event({
                "type": "card", "player": player.playerNumber, "card": specificCard
            })
            self.incrementTurn()
        else:
            print("Error")

        print("Player: ", player.playerNumber, " played: ", end=' ')
        print(specificCard)

        self.cleanCardFromPlayers(specificCard)

        # Restart the trick
        startingPoint = len(self.pile)
        for n in range(startingPoint,self.numberofplayers):
            player = self.players[ self.turn ]
            # Each player makes their move
            
            # Player makes their move
            recorder = getattr(self, "training_recorder", None)
            snapshot = self.observe(player.playerNumber) if recorder is not None else None
            legal = player.getRealPossibleMoves(self.pile) if recorder is not None else None
            cardPlayed = player.makeMove(self.pile.copy(), self.trumpsuit, self)
            if recorder is not None:
                recorder.on_decision(self, snapshot, legal, cardPlayed)

            self.pile.append( cardPlayed )
            self._record_public_event({
                "type": "card", "player": player.playerNumber, "card": cardPlayed
            })

            print("Player: ", player.playerNumber, " played: ", end=' ')
            print(cardPlayed)

            self.cleanCardFromPlayers(cardPlayed)
            
            self.incrementTurn()

        # See who has won and move on
        winningplay = self.extractTrickWinner()
        self.updateTricksWon(winningplay)
        self._record_public_event({
            "type": "trick_end",
            "winner": (self.leader + winningplay) % self.numberofplayers
        })
        print("#############################################")
        self.setTurnToWinner()
        self.trickNumber = self.trickNumber + 1
        print(self.tricksWon)
        print("############### End of trick ################")

    def cleanCardFromPlayers(self, card, playerToIgnore = 0):
        for playerTag in self.players:
            if playerTag.playerNumber != playerToIgnore:
                playerTag.removePossible(card)

    def setTurnToWinner(self):
        winningplay = self.extractTrickWinner()
        winningPlayerIndex = (self.leader + winningplay)% self.numberofplayers
        print("Player: ", winningPlayerIndex, " won the trick with so it is now: ", winningPlayerIndex, "'s' go")
        self.leader = winningPlayerIndex
        self.turn = self.leader


    def endRound(self):
        self._record_public_event({"type": "round_end", "scores": list(self.scores)})
        # Increment the round number and the dealer number
        self.trickNumber = 0
        self.roundnumber = self.roundnumber + 1
        self.dealer = (self.dealer + 1) % self.numberofplayers
        self.leader = (self.dealer + 1) % self.numberofplayers

    def extractTrickWinner(self):
        winningplay = analyse_pile(self.pile, self.trumpsuit)
        return winningplay

    def updateTricksWon(self, winningplay):
        winningPlayerIndex = (self.leader + winningplay)% self.numberofplayers
        self.tricksWon[winningPlayerIndex] = self.tricksWon[winningPlayerIndex] + 1

    def incrementTurn(self):
        print("------------------------------------------")
        print("Incrementing turn from ", end=' ')
        print(self.turn, end=' ')
        self.turn = ( self.turn + 1) % self.numberofplayers
        print(" to ", end=' ')
        print(self.turn)
        print("For tick number: ", self.trickNumber)
        
        print("------------------------------------------")

    def getPlayerBids(self):
        # Let the players make their bid
        bidno = self.dealer
        currentBids = []
        for n in range(0,self.numberofplayers):
            bidno = (bidno + 1) % self.numberofplayers
            player = self.players[bidno]
            ncards = self.cardnumbers[self.roundnumber]
            # Optional information-safe bid search: only give the bidding
            # player its own hand plus public bidding history, never Game.
            if hasattr(player, "makeBidWithObservation"):
                thisPlayerBid = player.makeBidWithObservation(
                    self.numberofplayers, ncards, self.trumpsuit,
                    currentBids.copy(), self.observe(bidno))
            else:
                thisPlayerBid = player.makeBid(
                    self.numberofplayers, ncards, self.trumpsuit,
                    currentBids.copy())
            currentBids.append(thisPlayerBid)
            self.bids[bidno] = thisPlayerBid
            self._record_public_event({
                "type": "bid", "player": bidno, "bid": thisPlayerBid
            })

    def resetRound(self):
        # Empty the tricks and the bids
        self.bids = []
        self.tricksWon = []
        self.trickNumber = 0
        for i in range(0,self.numberofplayers):
            self.bids.append(-1)
            self.tricksWon.append(0)

    def resetGame(self):
        # Transient per round
        self.pile = []
        self.bids = []
        self.scores = []
        for pIt in range(0,self.numberofplayers):
            self.scores.append(0)
        self.dealer = 0
        self.leader = (self.dealer + 1) % self.numberofplayers
        self.trumpsuit = 'h'
        self.roundnumber = 0
        self.trickNumber = 0
        self._information_state = None
        self._observer_cache = {}
        # Recorder is opt-in, never part of game-state serialization.
        self.training_recorder = None

    def manualDeal(self, cardsinround):
        """Use the entered first hand and deal non-overlapping opponent hands."""
        first_hand = list(generate_manual_deal(cardsinround))
        if len(first_hand) != cardsinround or len(set(first_hand)) != cardsinround:
            raise ValueError('Manual deal must contain the requested number of unique cards')
        if not set(first_hand).issubset(all_cards_in_deck):
            raise ValueError('Manual deal contains an invalid card')
        unseen = [card for card in all_cards_in_deck if card not in first_hand]
        if (self.numberofplayers - 1) * cardsinround > len(unseen):
            raise ValueError('Not enough cards for all players')
        sampled = random.sample(unseen, (self.numberofplayers - 1) * cardsinround)
        self.players[0].set_hands(first_hand)
        for i, player in enumerate(self.players[1:]):
            hand = sampled[i * cardsinround:(i + 1) * cardsinround]
            player.set_hands(hand, unseen)
        self._begin_observer_round(cardsinround)

    def randomDeal(self, cardsinround):
        """Deal independent real hands; only the observed hand is fully known."""
        player_hands = generate_random_deal(cardsinround, self.numberofplayers)
        if not isinstance(player_hands, list):
            raise ValueError('Not enough cards for all players')
        known = set(player_hands[0])
        unseen = [card for card in all_cards_in_deck if card not in known]
        for i, player in enumerate(self.players):
            player.set_hands(player_hands[i], player_hands[i] if i == 0 else unseen)
        self._begin_observer_round(cardsinround)

    def processScore(self):
        # Compare the number of tricks that each player has won 
        # with the number of tricks that they bid to win
        for playerIterator in range(0,self.numberofplayers):
            thisTrickCount = self.tricksWon[playerIterator]
            thisBid = self.bids[playerIterator]
            thisPlayerScore = tricksToScore(thisBid,thisTrickCount)
            self.scores[playerIterator] = self.scores[playerIterator] + thisPlayerScore
        recorder = getattr(self, "training_recorder", None)
        if recorder is not None:
            recorder.on_round_end(self)

    def _begin_observer_round(self, cards_per_player):
        """Start public inference without ever reading opponents' real hands."""
        previous = getattr(self, "_information_state", None)
        history = (previous.history_json if previous is not None else ())
        memory = previous.policy_state if previous is not None else None
        self._information_state = InformationState.new_round(
            own_hand=self.players[0].real_hand,
            num_players=self.numberofplayers,
            round_index=self.roundnumber,
            dealer=self.dealer,
            leader=self.leader,
            cards_per_player=cards_per_player,
            scores=self.scores,
            previous_history=history,
            policy_state=memory,
        )
        # Observer states are specific to one dealt round. Cached observers
        # must not carry real hands/knowledge across independent deals.
        self._observer_cache = {}
        recorder = getattr(self, "training_recorder", None)
        if recorder is not None:
            recorder.on_round_start(self)

    def _record_public_event(self, event):
        """Update beliefs using *only* the public event."""
        state = getattr(self, "_information_state", None)
        if state is not None:
            self._information_state = state.apply_public_event(event)

    def observe(self, player_id=0):
        """Return only this player's private hand and publicly known cards.

        Player zero has an incrementally updated InformationState. For
        other players, lazily reconstruct once per round and then apply only
        NEW public events. This avoids repeatedly replaying entire round and
        match history on every turn, which dominated training-game traces.

        Cache is invalidated at each deal and on a changed history prefix.
        Policy memory is attached fresh on every call (never cached).
        """
        state = getattr(self, "_information_state", None)
        if state is None:
            raise ValueError("Deal the round before observing")
        if not 0 <= player_id < self.numberofplayers:
            raise ValueError("Invalid observer player ID")

        if player_id == state.observer_id:
            observation = state
        else:
            if state.phase not in ("dealt", "bidding", "playing"):
                raise ValueError("Alternate observers require an active round")
            cache = getattr(self, "_observer_cache", None)
            if cache is None:
                cache = self._observer_cache = {}
            observation = cache.get(player_id)
            # Verify the public-history prefix before replaying just the
            # suffix. A caller may legitimately replace _information_state
            # during a deterministic simulation or test.
            if (observation is None
                    or len(observation.history_json) > len(state.history_json)
                    or observation.history_json != state.history_json[:len(observation.history_json)]):
                events = state.public_history
                first = max(i for i, event in enumerate(events)
                            if event["type"] == "round_start")
                round_events = events[first:]
                start_event = round_events[0]
                own_start = list(self.players[player_id].real_hand)
                own_start.extend(
                    event["card"] for event in round_events[1:]
                    if event["type"] == "card" and event["player"] == player_id
                )
                observation = InformationState.new_round(
                    own_hand=own_start,
                    num_players=self.numberofplayers,
                    round_index=start_event["round"],
                    dealer=start_event["dealer"],
                    leader=start_event["leader"],
                    cards_per_player=start_event["cards_per_player"],
                    scores=self.scores,
                    observer_id=player_id,
                    previous_history=state.history_json[:first],
                )
                for event in round_events[1:]:
                    observation = observation.apply_public_event(event)
            else:
                for event_json in state.history_json[len(observation.history_json):]:
                    observation = observation.apply_public_event(json.loads(event_json))
            cache[player_id] = observation

        player = self.players[player_id]
        return observation.with_policy_state(player.export_policy_state())

    @classmethod
    def from_information_state(cls, state, seed=0, player_factory=None):
        """Construct an independent, determinized card-decision rollout.

        No source Game is required. This supports states in the 'playing'
        phase, including partially played tricks. RNG used by policies
        during rollout must be seeded independently by the caller/worker.
        """
        if state.phase != "playing":
            raise ValueError("Rollout requires a playing-phase snapshot")
        from RandomPlayer import RandomPlayer
        hands = state.sample_hands(random.Random(seed))
        instance = cls.__new__(cls)
        schedule = [7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7]
        if state.round_index >= len(schedule) or (
            schedule[state.round_index] != state.cards_per_player
        ):
            schedule = [state.cards_per_player] * (state.round_index + 1)
        instance.cardnumbers = schedule
        instance.playerConfigFileName = ""
        instance.numberofplayers = state.num_players
        factory = player_factory or (lambda pid, hand: RandomPlayer(pid, list(hand)))
        instance.players = [factory(pid, hand) for pid, hand in enumerate(hands)]
        for player, hand in zip(instance.players, hands):
            player.set_hands(hand)
        zero_counters = [0] * state.num_players
        for event in state.public_history:
            if event["type"] == "bid":
                pid = event["player"]
                zero_counters[pid] = (
                    zero_counters[pid] + 1 if event["bid"] == 0 else 0
                )
        for pid, player in enumerate(instance.players):
            player.resetZeroCounter(zero_counters[pid])
        # Restore the observer policy's own JSON-compatible memory when that
        # policy is available. The factory may intentionally substitute a
        # different algorithm for counterfactual experiments.
        observer_policy = instance.players[state.observer_id]
        if state.policy_state is not None and hasattr(observer_policy, "load_policy_state"):
            observer_policy.load_policy_state(state.policy_state)
        instance.dealer = state.dealer
        instance.leader = state.leader
        instance.turn = state.turn
        instance.roundnumber = state.round_index
        instance.trickNumber = state.trick_number
        instance.trumpsuit = state.trump
        instance.pile = [card for _, card in state.pile]
        instance.bids = [(-1 if b is None else b) for b in state.bids]
        instance.tricksWon = list(state.tricks_won)
        instance.scores = list(state.scores)
        instance._information_state = state
        return instance

    def convertToDeterministicGame(self):
        """Sample mutually consistent hands for a Monte Carlo rollout.

        Player zero's hand is observed. Opponent hands are sampled from their
        possible-card sets with randomized bipartite matching, so the same
        card cannot occur in two hands. Check feasibility before mutation.
        """
        if not self.players:
            return
        known_hand = self.players[0].real_hand
        known = set(known_hand)
        if len(known) != len(known_hand) or not known.issubset(self.players[0].possible_hand):
            raise ValueError('Observed hand is inconsistent')

        slot_players = []
        options = []
        for player_index, player in enumerate(self.players[1:], start=1):
            candidates = [card for card in player.possible_hand if card not in known]
            if len(candidates) < player.n_cards_in_hand:
                raise ValueError('No consistent assignment of opponent hands')
            for _ in range(player.n_cards_in_hand):
                choices = candidates.copy()
                random.shuffle(choices)
                slot_players.append(player_index)
                options.append(choices)

        matching = {}  # card -> player-slot index

        def assign(slot, visited):
            for card in options[slot]:
                if card in visited:
                    continue
                visited.add(card)
                if card not in matching or assign(matching[card], visited):
                    matching[card] = slot
                    return True
            return False

        slots = list(range(len(options)))
        random.shuffle(slots)
        if any(not assign(slot, set()) for slot in slots):
            raise ValueError('No consistent assignment of opponent hands')

        hands = [[] for _ in self.players]
        hands[0] = known_hand
        for card, slot in matching.items():
            hands[slot_players[slot]].append(card)
        for player, hand in zip(self.players, hands):
            player.set_hands(hand)

