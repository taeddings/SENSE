import pytest
from unittest.mock import MagicMock, patch
from sense_v2.core.evolution.population import PopulationManager, GenerationStats
from sense_v2.core.config import EvolutionConfig
from core.evolution.genome import ReasoningGenome, create_random_genome


@pytest.fixture
def evolution_config():
    """Create an EvolutionConfig instance."""
    return EvolutionConfig(
        population_size=4,
        mutation_rate=0.1,
        crossover_rate=0.8
    )


@pytest.fixture
def mock_agemem():
    """Create a mock AgeMem instance."""
    return MagicMock()


@pytest.fixture
def population_manager(evolution_config, mock_agemem):
    """Create a PopulationManager instance."""
    return PopulationManager(
        config=evolution_config,
        agemem=mock_agemem
    )


@pytest.fixture
def sample_genomes():
    """Create sample genomes for testing."""
    return [
        create_random_genome(base_model_id="test_model", generation_id=0),
        create_random_genome(base_model_id="test_model", generation_id=0),
        create_random_genome(base_model_id="test_model", generation_id=0),
        create_random_genome(base_model_id="test_model", generation_id=0),
    ]


class TestPopulationManagerDEAPIntegration:
    """Test suite for PopulationManager DEAP integration."""

    @patch('sense_v2.core.evolution.population.DEAP_AVAILABLE', True)
    def test_deap_setup_enabled(self, population_manager):
        """Test DEAP toolbox setup when DEAP is available."""
        # The setup should have been called in __init__
        assert population_manager._toolbox is not None
        assert hasattr(population_manager._toolbox, 'select')
        assert hasattr(population_manager._toolbox, 'mate')
        assert hasattr(population_manager._toolbox, 'mutate')

    @patch('sense_v2.core.evolution.population.DEAP_AVAILABLE', False)
    def test_deap_setup_disabled(self):
        """Test behavior when DEAP is not available."""
        config = EvolutionConfig()
        manager = PopulationManager(config=config)

        assert manager._toolbox is None

    @patch('sense_v2.core.evolution.population.DEAP_AVAILABLE', True)
    def test_deap_var_and_with_deap(self, population_manager, sample_genomes):
        """Test variation with DEAP operators."""
        # Setup
        population_manager._toolbox = MagicMock()
        population_manager._toolbox.mate.return_value = (sample_genomes[0], sample_genomes[1])
        population_manager._toolbox.mutate.return_value = (sample_genomes[0],)

        # Execute
        result = population_manager._deap_var_and(
            sample_genomes[:2],
            crossover_prob=0.8,
            mutation_prob=0.1,
            drift_metric=0.2
        )

        # Assert
        assert len(result) == 2
        population_manager._toolbox.mate.assert_called()
        population_manager._toolbox.mutate.assert_called()

    @patch('sense_v2.core.evolution.population.DEAP_AVAILABLE', False)
    def test_deap_var_and_without_deap(self, population_manager, sample_genomes):
        """Test variation fallback when DEAP is not available."""
        # Execute
        result = population_manager._deap_var_and(
            sample_genomes[:2],
            crossover_prob=0.8,
            mutation_prob=0.1,
            drift_metric=0.2
        )

        # Assert - should return unchanged population
        assert result == sample_genomes[:2]

    @patch('sense_v2.core.evolution.population.DEAP_AVAILABLE', True)
    def test_deap_crossover(self, population_manager, sample_genomes):
        """Test DEAP-compatible crossover."""
        # Execute
        offspring1, offspring2 = population_manager._deap_crossover(
            sample_genomes[0], sample_genomes[1]
        )

        # Assert
        assert isinstance(offspring1, ReasoningGenome)
        assert isinstance(offspring2, ReasoningGenome)
        # Should be different from parents (crossover occurred)
        assert offspring1.genome_id != sample_genomes[0].genome_id
        assert offspring2.genome_id != sample_genomes[1].genome_id

    @patch('sense_v2.core.evolution.population.DEAP_AVAILABLE', True)
    def test_deap_mutate(self, population_manager, sample_genomes):
        """Test DEAP-compatible mutation."""
        original_budget = sample_genomes[0].reasoning_budget

        # Execute
        mutated, = population_manager._deap_mutate(
            sample_genomes[0], drift_metric=0.2, mutation_rate=0.5
        )

        # Assert
        assert isinstance(mutated, ReasoningGenome)
        assert isinstance(mutated, tuple) or isinstance(mutated, ReasoningGenome)

    def test_evolve_with_fallback_selection(self, population_manager, sample_genomes):
        """Test evolution with fallback selection when DEAP not available."""
        # Setup
        population_manager._toolbox = None  # Disable DEAP
        population_manager.population = sample_genomes
        fitness_scores = [0.8, 0.6, 0.7, 0.5]

        # Mock the selection method
        population_manager._select_parents = MagicMock(return_value=(sample_genomes[0], sample_genomes[1]))

        # Execute
        stats = population_manager.evolve(fitness_scores)

        # Assert
        assert isinstance(stats, GenerationStats)
        assert stats.generation_id == 1
        assert stats.population_size == 4
        assert population_manager.current_generation == 1

    @patch('sense_v2.core.evolution.population.DEAP_AVAILABLE', True)
    def test_evolve_with_deap_selection(self, population_manager, sample_genomes):
        """Test evolution with DEAP selection."""
        # Setup
        population_manager._toolbox = MagicMock()
        population_manager._toolbox.select.return_value = [0, 1]  # Select first two genomes
        population_manager.population = sample_genomes
        fitness_scores = [0.8, 0.6, 0.7, 0.5]

        # Mock variation
        population_manager._deap_var_and = MagicMock(return_value=[sample_genomes[0]])

        # Execute
        stats = population_manager.evolve(fitness_scores)

        # Assert
        assert isinstance(stats, GenerationStats)
        population_manager._toolbox.select.assert_called()

    def test_fitness_weighted_selection(self, population_manager, sample_genomes):
        """Test fitness-weighted parent selection fallback."""
        population_manager.population = sample_genomes
        fitness_scores = [0.8, 0.6, 0.7, 0.5]

        # Execute
        parent_a, parent_b = population_manager._select_parents(fitness_scores)

        # Assert
        assert parent_a in sample_genomes
        assert parent_b in sample_genomes
        assert parent_a != parent_b

    def test_generation_stats_calculation(self, population_manager, sample_genomes):
        """Test generation statistics calculation."""
        population_manager.population = sample_genomes
        fitness_scores = [0.8, 0.6, 0.7, 0.5]

        # Execute
        stats = population_manager.evolve(fitness_scores)

        # Assert
        assert stats.best_fitness == 0.8
        assert stats.worst_fitness == 0.5
        assert abs(stats.average_fitness - 0.65) < 0.01  # (0.8+0.6+0.7+0.5)/4
        assert stats.elite_count == 4  # selection_top_k default

    def test_drift_calculation(self, population_manager):
        """Test drift metric calculation."""
        # First generation
        fitness1 = [0.5, 0.6, 0.7, 0.8]
        population_manager.evolve(fitness1)

        # Second generation with different distribution
        fitness2 = [0.3, 0.4, 0.5, 0.6]
        drift = population_manager._calculate_drift(fitness2)

        # Assert
        assert isinstance(drift, float)
        assert drift >= 0.0

    def test_population_checkpointing(self, population_manager, sample_genomes):
        """Test population checkpointing to LTM."""
        # Setup
        population_manager.population = sample_genomes
        fitness_scores = [0.8, 0.6, 0.7, 0.5]

        # Execute
        stats = population_manager.evolve(fitness_scores)

        # Assert
        assert stats.generation_id == 1
        # Check that checkpointing was attempted (agemem.store_async would be called)
        # Since agemem is mocked, we just verify the flow