"""Scaling curve analysis and visualization.

This module provides:
- Scaling curve fitting (power law)
- Visualization of scaling experiments
- Comparison across checkpoints
"""

import json
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class ScalingPoint:
    """A single point on the scaling curve."""
    num_samples: int
    metric_value: float
    metric_name: str
    checkpoint_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingCurve:
    """A fitted scaling curve."""
    metric_name: str
    points: List[ScalingPoint]

    # Power law fit: y = a * x^b + c
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    r_squared: float = 0.0

    def predict(self, num_samples: int) -> float:
        """Predict metric value for given number of samples."""
        return self.a * (num_samples ** self.b) + self.c

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metric_name': self.metric_name,
            'points': [
                {
                    'num_samples': p.num_samples,
                    'metric_value': p.metric_value,
                    'checkpoint_name': p.checkpoint_name,
                }
                for p in self.points
            ],
            'fit': {
                'a': self.a,
                'b': self.b,
                'c': self.c,
                'r_squared': self.r_squared,
            }
        }


class ScalingAnalyzer:
    """Analyzes scaling experiments and fits curves.

    This class:
    1. Loads experiment results
    2. Fits power law curves
    3. Generates visualizations
    4. Predicts performance at larger scales
    """

    def __init__(self):
        """Initialize analyzer."""
        self.curves: Dict[str, ScalingCurve] = {}
        self.raw_data: List[Dict[str, Any]] = []

    def load_experiment(self, experiment_path: Path) -> Dict[str, Any]:
        """Load experiment from JSON file.

        Args:
            experiment_path: Path to experiment.json

        Returns:
            Experiment data
        """
        with open(experiment_path, 'r') as f:
            data = json.load(f)

        self.raw_data = []

        for checkpoint in data.get('checkpoints', []):
            if checkpoint.get('status') in ['completed', 'evaluated']:
                entry = {
                    'name': checkpoint['name'],
                    'num_samples': checkpoint['num_samples'],
                    'difficulty_distribution': checkpoint.get('difficulty_distribution', {}),
                    'final_loss': checkpoint.get('final_loss'),
                    **checkpoint.get('eval_results', {}),
                }
                self.raw_data.append(entry)

        print(f"Loaded {len(self.raw_data)} checkpoints from {experiment_path}")

        return data

    def load_from_data(self, data: List[Dict[str, Any]]):
        """Load data directly.

        Args:
            data: List of checkpoint data dictionaries
        """
        self.raw_data = data
        print(f"Loaded {len(self.raw_data)} data points")

    def fit_scaling_curve(
        self,
        metric_name: str,
        log_scale: bool = True,
    ) -> ScalingCurve:
        """Fit a power law scaling curve.

        Args:
            metric_name: Name of metric to fit (e.g., 'accuracy', 'final_loss')
            log_scale: Whether to fit in log-log space (for power law)

        Returns:
            Fitted ScalingCurve
        """
        # Extract points
        points = []
        for entry in self.raw_data:
            if metric_name in entry and entry[metric_name] is not None:
                points.append(ScalingPoint(
                    num_samples=entry['num_samples'],
                    metric_value=entry[metric_name],
                    metric_name=metric_name,
                    checkpoint_name=entry['name'],
                    metadata=entry,
                ))

        if len(points) < 2:
            print(f"Warning: Not enough points for fitting {metric_name}")
            curve = ScalingCurve(metric_name=metric_name, points=points)
            self.curves[metric_name] = curve
            return curve

        # Sort by num_samples
        points.sort(key=lambda p: p.num_samples)

        # Extract x, y values
        x = np.array([p.num_samples for p in points], dtype=float)
        y = np.array([p.metric_value for p in points], dtype=float)

        # Fit power law: y = a * x^b + c
        # In log-log space: log(y - c) = log(a) + b * log(x)
        # For simplicity, use linear regression in log-log space

        a, b, c = 1.0, 0.0, 0.0
        r_squared = 0.0

        if log_scale and np.all(y > 0) and np.all(x > 0):
            # Fit in log-log space
            log_x = np.log(x)
            log_y = np.log(y)

            # Linear regression
            n = len(x)
            sum_x = np.sum(log_x)
            sum_y = np.sum(log_y)
            sum_xy = np.sum(log_x * log_y)
            sum_xx = np.sum(log_x * log_x)

            # y = a + b*x in log space -> y = exp(a) * x^b in original space
            denom = n * sum_xx - sum_x * sum_x
            if abs(denom) > 1e-10:
                b = (n * sum_xy - sum_x * sum_y) / denom
                log_a = (sum_y - b * sum_x) / n
                a = np.exp(log_a)
                c = 0.0

                # R-squared
                y_pred = a * (x ** b)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            # Simple linear fit
            n = len(x)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_xx = np.sum(x * x)

            denom = n * sum_xx - sum_x * sum_x
            if abs(denom) > 1e-10:
                b = (n * sum_xy - sum_x * sum_y) / denom
                a = (sum_y - b * sum_x) / n
                c = 0.0

                # This is linear fit, convert to our format
                # y = a + b*x -> y = b*x + a
                a, b, c = b, 1.0, a

                # R-squared
                y_pred = a * x + c
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Create curve
        curve = ScalingCurve(
            metric_name=metric_name,
            points=points,
            a=float(a),
            b=float(b),
            c=float(c),
            r_squared=float(r_squared),
        )

        self.curves[metric_name] = curve

        print(f"\nFitted curve for {metric_name}:")
        print(f"  y = {a:.4f} * x^{b:.4f} + {c:.4f}")
        print(f"  R² = {r_squared:.4f}")

        return curve

    def predict_at_scale(
        self,
        metric_name: str,
        num_samples_list: List[int],
    ) -> List[Tuple[int, float]]:
        """Predict metric values at different scales.

        Args:
            metric_name: Name of metric
            num_samples_list: List of sample sizes to predict

        Returns:
            List of (num_samples, predicted_value) tuples
        """
        if metric_name not in self.curves:
            self.fit_scaling_curve(metric_name)

        curve = self.curves[metric_name]
        predictions = []

        for n in num_samples_list:
            pred = curve.predict(n)
            predictions.append((n, pred))
            print(f"  {n:,} samples -> {metric_name} = {pred:.4f}")

        return predictions

    def compare_checkpoints(self) -> Dict[str, Any]:
        """Compare all checkpoints.

        Returns:
            Comparison statistics
        """
        if not self.raw_data:
            return {}

        comparison = {
            'checkpoints': [],
            'improvement_rates': {},
        }

        # Get metrics present in data
        metrics = set()
        for entry in self.raw_data:
            for key, value in entry.items():
                if isinstance(value, (int, float)) and key not in ['num_samples']:
                    metrics.add(key)

        # Sort by num_samples
        sorted_data = sorted(self.raw_data, key=lambda x: x['num_samples'])

        for entry in sorted_data:
            comparison['checkpoints'].append({
                'name': entry['name'],
                'num_samples': entry['num_samples'],
                **{m: entry.get(m) for m in metrics if entry.get(m) is not None},
            })

        # Calculate improvement rates
        for metric in metrics:
            values = [(e['num_samples'], e.get(metric)) for e in sorted_data if e.get(metric) is not None]
            if len(values) >= 2:
                first_n, first_v = values[0]
                last_n, last_v = values[-1]

                if first_v != 0:
                    improvement = (last_v - first_v) / abs(first_v) * 100
                else:
                    improvement = float('inf') if last_v != 0 else 0

                comparison['improvement_rates'][metric] = {
                    'from': (first_n, first_v),
                    'to': (last_n, last_v),
                    'improvement_pct': improvement,
                }

        return comparison

    def get_summary(self) -> str:
        """Get text summary of analysis.

        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 60,
            "SCALING ANALYSIS SUMMARY",
            "=" * 60,
            "",
        ]

        # Checkpoints
        lines.append("CHECKPOINTS:")
        for entry in sorted(self.raw_data, key=lambda x: x['num_samples']):
            lines.append(f"  {entry['name']}: {entry['num_samples']:,} samples")
            for key, value in entry.items():
                if isinstance(value, float) and key not in ['num_samples']:
                    lines.append(f"    - {key}: {value:.4f}")
        lines.append("")

        # Fitted curves
        if self.curves:
            lines.append("FITTED CURVES:")
            for name, curve in self.curves.items():
                lines.append(f"  {name}:")
                lines.append(f"    y = {curve.a:.4f} * x^{curve.b:.4f} + {curve.c:.4f}")
                lines.append(f"    R² = {curve.r_squared:.4f}")
            lines.append("")

        # Comparison
        comparison = self.compare_checkpoints()
        if comparison.get('improvement_rates'):
            lines.append("IMPROVEMENT RATES:")
            for metric, info in comparison['improvement_rates'].items():
                lines.append(f"  {metric}:")
                lines.append(f"    {info['from'][0]:,} -> {info['to'][0]:,} samples")
                lines.append(f"    {info['from'][1]:.4f} -> {info['to'][1]:.4f}")
                lines.append(f"    Improvement: {info['improvement_pct']:.1f}%")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def save_analysis(self, output_path: Path):
        """Save analysis results to JSON.

        Args:
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        analysis = {
            'raw_data': self.raw_data,
            'curves': {name: curve.to_dict() for name, curve in self.curves.items()},
            'comparison': self.compare_checkpoints(),
        }

        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        print(f"Analysis saved to {output_path}")


def plot_scaling_curve(
    analyzer: ScalingAnalyzer,
    metric_name: str,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    log_scale: bool = True,
    show_predictions: Optional[List[int]] = None,
):
    """Plot scaling curve for a metric.

    Args:
        analyzer: ScalingAnalyzer with fitted curves
        metric_name: Name of metric to plot
        output_path: Optional path to save figure
        title: Optional title for plot
        log_scale: Whether to use log-log scale
        show_predictions: Optional list of sample sizes to show predictions for
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping plot")
        return

    if metric_name not in analyzer.curves:
        analyzer.fit_scaling_curve(metric_name, log_scale=log_scale)

    curve = analyzer.curves[metric_name]

    if not curve.points:
        print(f"No data points for {metric_name}")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data
    x = [p.num_samples for p in curve.points]
    y = [p.metric_value for p in curve.points]
    labels = [p.checkpoint_name for p in curve.points]

    # Plot actual points
    ax.scatter(x, y, s=100, c='blue', zorder=5, label='Actual')

    # Add labels
    for i, label in enumerate(labels):
        ax.annotate(
            label,
            (x[i], y[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9,
        )

    # Plot fitted curve
    if curve.r_squared > 0:
        x_range = np.linspace(min(x) * 0.8, max(x) * 1.5, 100)
        y_fit = curve.a * (x_range ** curve.b) + curve.c
        ax.plot(x_range, y_fit, 'r--', alpha=0.7, label=f'Fit (R²={curve.r_squared:.3f})')

    # Plot predictions
    if show_predictions:
        predictions = analyzer.predict_at_scale(metric_name, show_predictions)
        x_pred = [p[0] for p in predictions]
        y_pred = [p[1] for p in predictions]
        ax.scatter(x_pred, y_pred, s=80, c='green', marker='^', zorder=4, label='Predictions')

        for i, (xp, yp) in enumerate(predictions):
            ax.annotate(
                f'{xp:,}',
                (xp, yp),
                textcoords="offset points",
                xytext=(0, -15),
                ha='center',
                fontsize=8,
                color='green',
            )

    # Formatting
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.set_xlabel('Number of Training Samples', fontsize=12)
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title or f'Scaling Curve: {metric_name}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_difficulty_breakdown(
    analyzer: ScalingAnalyzer,
    output_path: Optional[Path] = None,
):
    """Plot difficulty distribution across checkpoints.

    Args:
        analyzer: ScalingAnalyzer with data
        output_path: Optional path to save figure
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping plot")
        return

    if not analyzer.raw_data:
        print("No data to plot")
        return

    # Prepare data
    sorted_data = sorted(analyzer.raw_data, key=lambda x: x['num_samples'])

    names = [d['name'] for d in sorted_data]
    hard = [d.get('difficulty_distribution', {}).get('hard', 0) for d in sorted_data]
    medium = [d.get('difficulty_distribution', {}).get('medium', 0) for d in sorted_data]
    easy = [d.get('difficulty_distribution', {}).get('easy', 0) for d in sorted_data]

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(names))
    width = 0.6

    ax.bar(x, hard, width, label='Hard', color='#e74c3c')
    ax.bar(x, medium, width, bottom=hard, label='Medium', color='#f39c12')
    ax.bar(x, easy, width, bottom=[h + m for h, m in zip(hard, medium)], label='Easy', color='#27ae60')

    # Add total labels
    for i, (h, m, e) in enumerate(zip(hard, medium, easy)):
        total = h + m + e
        ax.annotate(
            f'{total:,}',
            (i, total),
            textcoords="offset points",
            xytext=(0, 5),
            ha='center',
            fontweight='bold',
        )

    ax.set_xlabel('Checkpoint', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Difficulty Distribution by Checkpoint', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()
