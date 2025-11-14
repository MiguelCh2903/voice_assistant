"""
Finite State Machine implementation for voice assistant core.

This module implements a robust state machine to manage the voice assistant's
operational states and transitions, including error handling and recovery.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional

from ..audio.types import AssistantState
from ..constants import (
    MAX_ERROR_RECOVERY_ATTEMPTS,
    STATE_TRANSITION_TIMEOUT_SEC,
)


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    pass


class StateMachineTimeoutError(Exception):
    """Raised when a state transition times out."""

    pass


class VoiceAssistantStateMachine:
    """
    Finite State Machine for voice assistant core operations.

    Manages state transitions, validates transitions, handles timeouts,
    and provides error recovery mechanisms.
    """

    # Valid state transitions mapping
    # Valid transitions follow the Home Assistant assist_pipeline pattern
    # Supports continuous streaming with interruptions and re-entry from any state
    VALID_TRANSITIONS = {
        AssistantState.DISCONNECTED: {AssistantState.CONNECTING, AssistantState.ERROR},
        AssistantState.CONNECTING: {
            AssistantState.IDLE,
            AssistantState.DISCONNECTED,
            AssistantState.ERROR,
        },
        AssistantState.IDLE: {
            AssistantState.STREAMING_AUDIO,
            AssistantState.DISCONNECTED,
            AssistantState.ERROR,
        },
        AssistantState.STREAMING_AUDIO: {
            AssistantState.TRANSCRIBING,
            AssistantState.IDLE,
            AssistantState.ERROR,
        },
        AssistantState.TRANSCRIBING: {
            AssistantState.PROCESSING_LLM,
            AssistantState.STREAMING_AUDIO,  # Allows re-entry if the user continues speaking
            AssistantState.IDLE,
            AssistantState.ERROR,
        },
        AssistantState.PROCESSING_LLM: {
            AssistantState.PLAYING_RESPONSE,
            AssistantState.STREAMING_AUDIO,  # Allows returning to streaming if interrupted during LLM processing
            AssistantState.IDLE,
            AssistantState.ERROR,
        },
        AssistantState.PLAYING_RESPONSE: {
            AssistantState.STREAMING_AUDIO,  # Allows interruptions while the response is playing
            AssistantState.IDLE,
            AssistantState.ERROR,
        },
        AssistantState.ERROR: {
            AssistantState.DISCONNECTED,
            AssistantState.IDLE,
            AssistantState.CONNECTING,
        },
    }

    def __init__(self, initial_state: AssistantState = AssistantState.DISCONNECTED):
        """
        Initialize the state machine.

        Args:
            initial_state: Initial state for the state machine
        """
        self._current_state = initial_state
        self._previous_state = None
        self._state_entry_time = time.time()
        self._state_history = [initial_state]
        self._error_count = 0
        self._last_error_time = 0.0
        self._recovery_attempts = 0

        # Callbacks for state transitions
        self._state_entry_callbacks: Dict[AssistantState, Callable] = {}
        self._state_exit_callbacks: Dict[AssistantState, Callable] = {}
        self._transition_callbacks: Dict[tuple, Callable] = {}

        # Logger
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self._logger.info(f"State machine initialized in state: {initial_state.name}")

    @property
    def current_state(self) -> AssistantState:
        """Get the current state."""
        return self._current_state

    @property
    def previous_state(self) -> Optional[AssistantState]:
        """Get the previous state."""
        return self._previous_state

    @property
    def time_in_current_state(self) -> float:
        """Get time spent in current state in seconds."""
        return time.time() - self._state_entry_time

    @property
    def error_count(self) -> int:
        """Get total error count."""
        return self._error_count

    @property
    def state_history(self) -> list:
        """Get state transition history."""
        return self._state_history.copy()

    def is_valid_transition(self, target_state: AssistantState) -> bool:
        """
        Check if transition to target state is valid.

        Args:
            target_state: State to transition to

        Returns:
            True if transition is valid, False otherwise
        """
        valid_targets = self.VALID_TRANSITIONS.get(self._current_state, set())
        return target_state in valid_targets

    def transition_to(
        self,
        target_state: AssistantState,
        event_data: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> bool:
        """
        Transition to a new state.

        Args:
            target_state: State to transition to
            event_data: Optional data associated with the transition
            force: Force transition even if not normally valid

        Returns:
            True if transition successful, False otherwise

        Raises:
            StateTransitionError: If transition is invalid and not forced
        """
        if target_state == self._current_state:
            self._logger.debug(f"Already in target state: {target_state.name}")
            return True

        if not force and not self.is_valid_transition(target_state):
            error_msg = (
                f"Invalid transition from {self._current_state.name} "
                f"to {target_state.name}"
            )
            self._logger.error(error_msg)
            raise StateTransitionError(error_msg)

        # Execute exit callback for current state
        if self._current_state in self._state_exit_callbacks:
            try:
                self._state_exit_callbacks[self._current_state](
                    self._current_state, target_state, event_data
                )
            except Exception as e:
                self._logger.error(f"Error in state exit callback: {e}")

        # Execute transition callback
        transition_key = (self._current_state, target_state)
        if transition_key in self._transition_callbacks:
            try:
                self._transition_callbacks[transition_key](
                    self._current_state, target_state, event_data
                )
            except Exception as e:
                self._logger.error(f"Error in transition callback: {e}")

        # Update state
        self._previous_state = self._current_state
        self._current_state = target_state
        self._state_entry_time = time.time()
        self._state_history.append(target_state)

        # Limit history size
        if len(self._state_history) > 100:
            self._state_history = self._state_history[-50:]

        # Handle error state
        if target_state == AssistantState.ERROR:
            self._error_count += 1
            self._last_error_time = time.time()

        # Reset recovery attempts on successful transition to non-error state
        if (
            target_state != AssistantState.ERROR
            and self._previous_state == AssistantState.ERROR
        ):
            self._recovery_attempts = 0

        self._logger.info(
            f"State transition: {self._previous_state.name} -> {target_state.name}"
        )

        # Execute entry callback for new state
        if target_state in self._state_entry_callbacks:
            try:
                callback = self._state_entry_callbacks[target_state]
                # Check if callback is a coroutine and handle appropriately
                if asyncio.iscoroutinefunction(callback):
                    # Schedule the coroutine to run in the event loop
                    task = asyncio.create_task(
                        callback(self._previous_state, target_state, event_data)
                    )
                    # Don't wait for it to complete, just let it run
                    task.add_done_callback(
                        lambda t: self._handle_callback_completion(t, target_state)
                    )
                else:
                    # Call synchronous callback directly
                    callback(self._previous_state, target_state, event_data)
            except Exception as e:
                self._logger.error(f"Error in state entry callback: {e}")

        return True

    def handle_error(
        self,
        error_code: int,
        error_message: str = "",
        event_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Handle an error condition and transition to ERROR state.

        Args:
            error_code: Error code from ErrorCodes
            error_message: Human readable error message
            event_data: Additional error context data

        Returns:
            True if error handled successfully
        """
        if event_data is None:
            event_data = {}

        event_data.update(
            {
                "error_code": error_code,
                "error_message": error_message,
                "previous_state": self._current_state.name,
                "time_in_state": self.time_in_current_state,
            }
        )

        self._logger.error(
            f"Error occurred in state {self._current_state.name}: "
            f"Code {error_code}, Message: {error_message}"
        )

        return self.transition_to(AssistantState.ERROR, event_data, force=True)

    def attempt_recovery(self) -> bool:
        """
        Attempt to recover from ERROR state.

        Returns:
            True if recovery attempted, False if max attempts exceeded
        """
        if self._current_state != AssistantState.ERROR:
            self._logger.warning("Recovery attempted from non-error state")
            return False

        if self._recovery_attempts >= MAX_ERROR_RECOVERY_ATTEMPTS:
            self._logger.error("Maximum recovery attempts exceeded")
            return False

        self._recovery_attempts += 1

        # Determine recovery target state based on error context and history
        recovery_state = self._determine_recovery_state()

        self._logger.info(
            f"Attempting recovery #{self._recovery_attempts} "
            f"to state: {recovery_state.name}"
        )

        try:
            return self.transition_to(
                recovery_state,
                {
                    "recovery_attempt": self._recovery_attempts,
                    "recovery_from_error": True,
                },
            )
        except StateTransitionError:
            self._logger.error(f"Recovery transition to {recovery_state.name} failed")
            return False

    def _determine_recovery_state(self) -> AssistantState:
        """
        Determine the appropriate state for error recovery.

        Returns:
            Target state for recovery
        """
        # Check recent state history to determine best recovery state
        if len(self._state_history) >= 2:
            # Get the state before ERROR
            pre_error_state = self._state_history[-2]

            # Recovery logic based on where error occurred
            if pre_error_state in [
                AssistantState.STREAMING_AUDIO,
                AssistantState.TRANSCRIBING,
                AssistantState.PROCESSING_LLM,
                AssistantState.PLAYING_RESPONSE,
            ]:
                # Errors during processing - go back to IDLE
                return AssistantState.IDLE
            elif pre_error_state == AssistantState.CONNECTING:
                # Connection error - try disconnected state
                return AssistantState.DISCONNECTED
            elif pre_error_state == AssistantState.IDLE:
                # Error from idle - try to maintain connection
                return AssistantState.IDLE

        # Default recovery - try to reconnect
        return AssistantState.DISCONNECTED

    def is_timeout_exceeded(
        self, timeout_seconds: float = STATE_TRANSITION_TIMEOUT_SEC
    ) -> bool:
        """
        Check if current state has exceeded timeout.

        Args:
            timeout_seconds: Timeout threshold in seconds

        Returns:
            True if timeout exceeded
        """
        return self.time_in_current_state > timeout_seconds

    def register_state_entry_callback(
        self, state: AssistantState, callback: Callable
    ) -> None:
        """
        Register callback for state entry.

        Args:
            state: State to register callback for
            callback: Function to call on state entry
                     Signature: (previous_state, current_state, event_data)
        """
        self._state_entry_callbacks[state] = callback
        self._logger.debug(f"Registered entry callback for state: {state.name}")

    def register_state_exit_callback(
        self, state: AssistantState, callback: Callable
    ) -> None:
        """
        Register callback for state exit.

        Args:
            state: State to register callback for
            callback: Function to call on state exit
                     Signature: (current_state, target_state, event_data)
        """
        self._state_exit_callbacks[state] = callback
        self._logger.debug(f"Registered exit callback for state: {state.name}")

    def register_transition_callback(
        self, from_state: AssistantState, to_state: AssistantState, callback: Callable
    ) -> None:
        """
        Register callback for specific state transition.

        Args:
            from_state: Source state
            to_state: Target state
            callback: Function to call during transition
                     Signature: (from_state, to_state, event_data)
        """
        transition_key = (from_state, to_state)
        self._transition_callbacks[transition_key] = callback
        self._logger.debug(
            f"Registered transition callback: {from_state.name} -> {to_state.name}"
        )

    def reset(
        self, initial_state: AssistantState = AssistantState.DISCONNECTED
    ) -> None:
        """
        Reset the state machine to initial state.

        Args:
            initial_state: State to reset to
        """
        self._logger.info("Resetting state machine")

        self._previous_state = self._current_state
        self._current_state = initial_state
        self._state_entry_time = time.time()
        self._state_history = [initial_state]
        self._error_count = 0
        self._last_error_time = 0.0
        self._recovery_attempts = 0

        self._logger.info(f"State machine reset to: {initial_state.name}")

    def get_state_info(self) -> Dict[str, Any]:
        """
        Get comprehensive state machine information.

        Returns:
            Dictionary with state machine status
        """
        return {
            "current_state": self._current_state.name,
            "previous_state": self._previous_state.name
            if self._previous_state
            else None,
            "time_in_current_state": self.time_in_current_state,
            "state_entry_time": self._state_entry_time,
            "error_count": self._error_count,
            "last_error_time": self._last_error_time,
            "recovery_attempts": self._recovery_attempts,
            "state_history_length": len(self._state_history),
            "recent_states": [s.name for s in self._state_history[-5:]],
        }

    def _handle_callback_completion(
        self, task: asyncio.Task, state: AssistantState
    ) -> None:
        """Handle completion of async callback task."""
        try:
            if task.exception():
                self._logger.error(
                    f"Async callback for state {state.name} failed: {task.exception()}"
                )
        except Exception as e:
            self._logger.error(f"Error handling callback completion: {e}")
