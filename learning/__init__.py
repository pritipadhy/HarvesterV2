"""
Harvester v2 Learning Pipeline

Enables continuous improvement through outcome collection and training:

Components:
    - OutcomeCollector: Record action outcomes for feedback loops
    - PlaybookUpdater: Update playbook success rates based on outcomes
    - RewardCalculator: Calculate rewards for reinforcement learning
    - TrainingDataGenerator: Generate DPO pairs for fine-tuning

Learning flow:
    1. Actions are taken based on NBA recommendations
    2. Outcomes are collected (accepted, rejected, ignored)
    3. Rewards are calculated based on business impact
    4. Playbook success rates are updated
    5. Training data is generated for model improvement
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harvester_v2.learning.outcome_collector import OutcomeCollector
    from harvester_v2.learning.playbook_updater import PlaybookUpdater
    from harvester_v2.learning.reward_calculator import RewardCalculator
    from harvester_v2.learning.training_data_generator import TrainingDataGenerator
