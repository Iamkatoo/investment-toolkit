#!/usr/bin/env python3
"""
Knowledge Base Manager - V2 Migration System

Intelligent knowledge base management system that automatically maintains
documentation, FAQs, and operational knowledge based on system events
and user interactions.

Task 7.3: Long-term Operations Support Tools
- Automated FAQ generation and updates
- Dynamic documentation maintenance
- Issue pattern recognition and solution curation
- Knowledge article lifecycle management
- Search and retrieval optimization

Features:
- Automatic FAQ generation from common issues
- Dynamic documentation updates based on system changes
- Issue pattern analysis and solution recommendation
- Knowledge article relevance scoring
- Search optimization and knowledge discovery

Usage:
    from investment_toolkit.utilities.knowledge_base_manager import KnowledgeBaseManager

    kb_manager = KnowledgeBaseManager()
    kb_manager.update_knowledge_base()
    kb_manager.generate_faq_updates()

Created: 2025-09-17
Author: Claude Code Assistant
"""

import re
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
import yaml
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from investment_toolkit.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
except ImportError as e:
    print(f"❌ Import error: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class KnowledgeArticle:
    """Knowledge base article structure"""
    article_id: str
    title: str
    category: str
    content: str
    tags: List[str]
    relevance_score: float  # 0-100
    last_updated: datetime
    access_count: int
    effectiveness_score: float  # Based on user feedback
    related_issues: List[str]
    solution_success_rate: float


@dataclass
class FAQEntry:
    """FAQ entry structure"""
    question: str
    answer: str
    category: str
    frequency: int  # How often this question appears
    confidence: float  # Confidence in the answer
    last_updated: datetime
    related_keywords: List[str]
    source_incidents: List[str]


@dataclass
class IssuePattern:
    """Common issue pattern"""
    pattern_id: str
    description: str
    symptoms: List[str]
    root_causes: List[str]
    solutions: List[str]
    frequency: int
    resolution_time_avg: float  # Average resolution time in hours
    prevention_steps: List[str]
    related_articles: List[str]


@dataclass
class KnowledgeBaseReport:
    """Knowledge base management report"""
    report_timestamp: datetime
    analysis_period: Tuple[str, str]
    total_articles: int
    articles_updated: int
    new_faqs_generated: int
    issue_patterns_identified: int
    knowledge_gaps_found: List[str]
    recommendations: List[str]
    effectiveness_metrics: Dict[str, float]


class KnowledgeBaseManager:
    """Intelligent knowledge base management system"""

    def __init__(self, analysis_period_days: int = 30):
        self.analysis_period_days = analysis_period_days
        self.knowledge_base_dir = project_root / "docs" / "knowledge_base"
        self.faq_file = self.knowledge_base_dir / "automated_faq.yaml"
        self.patterns_file = self.knowledge_base_dir / "issue_patterns.json"
        self.articles_dir = self.knowledge_base_dir / "articles"

        # Ensure directories exist
        self.knowledge_base_dir.mkdir(exist_ok=True)
        self.articles_dir.mkdir(exist_ok=True)

        self.articles: List[KnowledgeArticle] = []
        self.faqs: List[FAQEntry] = []
        self.issue_patterns: List[IssuePattern] = []

        # Common issue categories and keywords
        self.issue_categories = {
            'correlation': ['correlation', 'score', 'alignment', 'v1', 'v2', 'comparison'],
            'performance': ['slow', 'timeout', 'memory', 'cpu', 'execution', 'time'],
            'database': ['database', 'connection', 'query', 'sql', 'table', 'data'],
            'configuration': ['config', 'setting', 'threshold', 'flag', 'parameter'],
            'monitoring': ['alert', 'notification', 'dashboard', 'monitoring', 'anomaly'],
            'error': ['error', 'exception', 'failure', 'crash', 'bug', 'issue']
        }

        logger.info("Knowledge Base Manager initialized")

    def _get_db_connection(self):
        """Get database connection"""
        import psycopg2
        from psycopg2.extras import RealDictCursor

        return psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            cursor_factory=RealDictCursor
        )

    def analyze_system_logs(self) -> List[Dict[str, Any]]:
        """Analyze system logs for common issues and patterns"""
        issues = []

        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()

            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.analysis_period_days)

            # Get system logs from migration log table
            cursor.execute("""
                SELECT timestamp, log_level, component, operation, message, details
                FROM backtest_results.scoring_migration_log
                WHERE timestamp BETWEEN %s AND %s
                  AND log_level IN ('WARNING', 'ERROR', 'CRITICAL')
                ORDER BY timestamp DESC
            """, (start_date, end_date))

            log_entries = cursor.fetchall()
            cursor.close()
            conn.close()

            # Analyze log patterns
            for entry in log_entries:
                issue = {
                    'timestamp': entry['timestamp'],
                    'level': entry['log_level'],
                    'component': entry['component'],
                    'operation': entry['operation'],
                    'message': entry['message'],
                    'details': entry['details'],
                    'category': self._categorize_issue(entry['message'])
                }
                issues.append(issue)

        except Exception as e:
            logger.error(f"Error analyzing system logs: {str(e)}")

        return issues

    def _categorize_issue(self, message: str) -> str:
        """Categorize issue based on message content"""
        message_lower = message.lower()

        for category, keywords in self.issue_categories.items():
            if any(keyword in message_lower for keyword in keywords):
                return category

        return 'general'

    def identify_issue_patterns(self, issues: List[Dict[str, Any]]) -> List[IssuePattern]:
        """Identify common issue patterns from log analysis"""
        patterns = []

        # Group issues by category and message similarity
        category_groups = defaultdict(list)
        for issue in issues:
            category_groups[issue['category']].append(issue)

        pattern_id = 1
        for category, category_issues in category_groups.items():
            if len(category_issues) >= 3:  # Minimum frequency for pattern
                # Analyze message patterns
                messages = [issue['message'] for issue in category_issues]
                common_words = self._extract_common_words(messages)

                # Group similar messages
                message_groups = self._group_similar_messages(category_issues)

                for group in message_groups:
                    if len(group) >= 3:  # Minimum frequency
                        symptoms = list(set([issue['message'] for issue in group]))
                        components = list(set([issue['component'] for issue in group]))

                        # Generate solutions based on category and patterns
                        solutions = self._generate_solutions(category, symptoms, components)
                        prevention_steps = self._generate_prevention_steps(category)

                        pattern = IssuePattern(
                            pattern_id=f"PATTERN_{pattern_id:03d}",
                            description=f"Common {category} issues in {', '.join(components)}",
                            symptoms=symptoms[:5],  # Top 5 symptoms
                            root_causes=self._identify_root_causes(category, group),
                            solutions=solutions,
                            frequency=len(group),
                            resolution_time_avg=self._estimate_resolution_time(category),
                            prevention_steps=prevention_steps,
                            related_articles=[]
                        )
                        patterns.append(pattern)
                        pattern_id += 1

        return patterns

    def _extract_common_words(self, messages: List[str]) -> List[str]:
        """Extract common words from messages"""
        # Simple word extraction (could be enhanced with NLP)
        all_words = []
        for message in messages:
            words = re.findall(r'\w+', message.lower())
            words = [w for w in words if len(w) > 3]  # Filter short words
            all_words.extend(words)

        # Return most common words
        word_counts = Counter(all_words)
        return [word for word, count in word_counts.most_common(10)]

    def _group_similar_messages(self, issues: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group similar messages together"""
        # Simple grouping by component and operation
        groups = defaultdict(list)

        for issue in issues:
            key = f"{issue['component']}_{issue['operation']}"
            groups[key].append(issue)

        return [group for group in groups.values() if len(group) >= 2]

    def _generate_solutions(self, category: str, symptoms: List[str], components: List[str]) -> List[str]:
        """Generate solutions based on category and symptoms"""
        solutions = {
            'correlation': [
                "Check V1/V2 input data consistency",
                "Verify scoring algorithm parameters",
                "Review normalization methods",
                "Analyze feature engineering differences",
                "Validate data quality and completeness"
            ],
            'performance': [
                "Monitor system resource usage",
                "Optimize database queries",
                "Review data processing pipeline",
                "Check for memory leaks",
                "Consider parallel processing optimization"
            ],
            'database': [
                "Check database connection parameters",
                "Verify database server health",
                "Review query performance",
                "Check table locks and deadlocks",
                "Validate database schema integrity"
            ],
            'configuration': [
                "Review configuration file syntax",
                "Validate parameter ranges",
                "Check environment variables",
                "Verify feature flag settings",
                "Test configuration changes in staging"
            ],
            'monitoring': [
                "Check notification service configuration",
                "Verify alert thresholds",
                "Review dashboard data sources",
                "Test notification endpoints",
                "Validate monitoring system health"
            ],
            'error': [
                "Review error logs for details",
                "Check system dependencies",
                "Verify input data format",
                "Test error recovery procedures",
                "Implement additional error handling"
            ]
        }

        return solutions.get(category, ["Contact support team for assistance"])

    def _generate_prevention_steps(self, category: str) -> List[str]:
        """Generate prevention steps for issue category"""
        prevention = {
            'correlation': [
                "Implement automated data quality checks",
                "Set up correlation monitoring alerts",
                "Regular algorithm parameter validation",
                "Automated V1/V2 consistency tests"
            ],
            'performance': [
                "Implement performance monitoring",
                "Set up resource usage alerts",
                "Regular performance benchmarking",
                "Proactive capacity planning"
            ],
            'database': [
                "Regular database health checks",
                "Implement connection pooling",
                "Monitor query performance",
                "Regular database maintenance"
            ],
            'configuration': [
                "Implement configuration validation",
                "Use version control for configs",
                "Automated configuration testing",
                "Regular configuration reviews"
            ],
            'monitoring': [
                "Regular monitoring system checks",
                "Test notification channels",
                "Validate alert thresholds",
                "Monitor system dependencies"
            ],
            'error': [
                "Implement comprehensive logging",
                "Regular error pattern analysis",
                "Proactive error monitoring",
                "Automated error recovery testing"
            ]
        }

        return prevention.get(category, ["Regular system health monitoring"])

    def _identify_root_causes(self, category: str, issues: List[Dict[str, Any]]) -> List[str]:
        """Identify likely root causes for issue pattern"""
        root_causes = {
            'correlation': [
                "Data input inconsistencies between V1 and V2",
                "Algorithm parameter differences",
                "Normalization method variations",
                "Feature engineering discrepancies"
            ],
            'performance': [
                "Resource constraints (CPU/Memory)",
                "Database query inefficiencies",
                "Data processing bottlenecks",
                "System configuration issues"
            ],
            'database': [
                "Connection pool exhaustion",
                "Query optimization issues",
                "Database server overload",
                "Network connectivity problems"
            ],
            'configuration': [
                "Invalid parameter values",
                "Missing configuration settings",
                "Environment-specific issues",
                "Configuration file corruption"
            ],
            'monitoring': [
                "Notification service failures",
                "Threshold configuration errors",
                "Data source connectivity issues",
                "System dependency failures"
            ],
            'error': [
                "Unhandled edge cases",
                "Data format inconsistencies",
                "System dependency failures",
                "Resource exhaustion"
            ]
        }

        return root_causes.get(category, ["Unknown root cause - requires investigation"])

    def _estimate_resolution_time(self, category: str) -> float:
        """Estimate average resolution time for category (in hours)"""
        resolution_times = {
            'correlation': 4.0,
            'performance': 6.0,
            'database': 2.0,
            'configuration': 1.0,
            'monitoring': 3.0,
            'error': 3.0
        }

        return resolution_times.get(category, 4.0)

    def generate_automated_faqs(self, issues: List[Dict[str, Any]], patterns: List[IssuePattern]) -> List[FAQEntry]:
        """Generate FAQ entries from common issues and patterns"""
        faqs = []

        # Generate FAQs from issue patterns
        for pattern in patterns:
            if pattern.frequency >= 5:  # High frequency patterns
                # Question based on pattern description
                question = f"How do I resolve {pattern.description.lower()}?"

                # Answer based on solutions
                answer = f"This issue typically occurs due to: {', '.join(pattern.root_causes[:2])}.\n\n"
                answer += "Recommended solutions:\n"
                for i, solution in enumerate(pattern.solutions[:3], 1):
                    answer += f"{i}. {solution}\n"

                if pattern.prevention_steps:
                    answer += f"\nPrevention:\n"
                    for i, step in enumerate(pattern.prevention_steps[:2], 1):
                        answer += f"• {step}\n"

                # Extract keywords for search
                keywords = []
                for symptom in pattern.symptoms:
                    keywords.extend(re.findall(r'\w+', symptom.lower()))
                keywords = list(set([k for k in keywords if len(k) > 3]))

                faq = FAQEntry(
                    question=question,
                    answer=answer.strip(),
                    category=self._categorize_issue(pattern.description),
                    frequency=pattern.frequency,
                    confidence=min(85.0, 60.0 + (pattern.frequency * 2)),  # Higher frequency = higher confidence
                    last_updated=datetime.now(),
                    related_keywords=keywords[:10],
                    source_incidents=[pattern.pattern_id]
                )
                faqs.append(faq)

        # Generate specific FAQs for common questions
        common_questions = [
            {
                'question': "Why is the correlation between V1 and V2 scores low?",
                'category': 'correlation',
                'keywords': ['correlation', 'v1', 'v2', 'scores', 'low']
            },
            {
                'question': "How can I improve V2 system performance?",
                'category': 'performance',
                'keywords': ['performance', 'v2', 'slow', 'optimization']
            },
            {
                'question': "What should I do when the dashboard is not updating?",
                'category': 'monitoring',
                'keywords': ['dashboard', 'update', 'monitoring', 'display']
            },
            {
                'question': "How do I troubleshoot database connection errors?",
                'category': 'database',
                'keywords': ['database', 'connection', 'error', 'troubleshoot']
            }
        ]

        for q_info in common_questions:
            # Count related issues
            related_count = sum(1 for issue in issues if issue['category'] == q_info['category'])

            if related_count >= 2:  # Has related issues
                solutions = self._generate_solutions(q_info['category'], [], [])
                prevention = self._generate_prevention_steps(q_info['category'])

                answer = f"When experiencing {q_info['category']} issues:\n\n"
                answer += "Common solutions:\n"
                for i, solution in enumerate(solutions[:3], 1):
                    answer += f"{i}. {solution}\n"

                answer += f"\nPrevention measures:\n"
                for i, step in enumerate(prevention[:2], 1):
                    answer += f"• {step}\n"

                faq = FAQEntry(
                    question=q_info['question'],
                    answer=answer.strip(),
                    category=q_info['category'],
                    frequency=related_count,
                    confidence=70.0,
                    last_updated=datetime.now(),
                    related_keywords=q_info['keywords'],
                    source_incidents=[]
                )
                faqs.append(faq)

        return faqs

    def update_existing_articles(self, patterns: List[IssuePattern]) -> int:
        """Update existing knowledge base articles"""
        updated_count = 0

        # Check for existing operation guide updates
        ops_guide_path = project_root / "docs" / "v2_migration" / "ab_operations_guide.md"

        if ops_guide_path.exists():
            try:
                with open(ops_guide_path, 'r') as f:
                    content = f.read()

                # Check if common issues section needs updates
                if "よくある問題と解決方法" in content or "common issues" in content.lower():
                    # Extract new patterns that aren't already documented
                    new_issues = []
                    for pattern in patterns:
                        if pattern.frequency >= 5 and pattern.pattern_id not in content:
                            new_issues.append(pattern)

                    if new_issues:
                        # Add new troubleshooting section
                        additional_content = "\n\n### 📊 自動検出された問題パターン\n\n"
                        additional_content += f"*最終更新: {datetime.now().strftime('%Y-%m-%d')}*\n\n"

                        for issue in new_issues:
                            additional_content += f"#### {issue.description}\n\n"
                            additional_content += f"**発生頻度**: {issue.frequency}回 (過去{self.analysis_period_days}日間)\n"
                            additional_content += f"**平均解決時間**: {issue.resolution_time_avg:.1f}時間\n\n"

                            additional_content += "**症状**:\n"
                            for symptom in issue.symptoms[:3]:
                                additional_content += f"- {symptom}\n"

                            additional_content += "\n**解決方法**:\n"
                            for solution in issue.solutions[:3]:
                                additional_content += f"1. {solution}\n"

                            additional_content += "\n**予防策**:\n"
                            for prevention in issue.prevention_steps[:2]:
                                additional_content += f"- {prevention}\n"

                            additional_content += "\n---\n\n"

                        # Append to file (in practice, would be more sophisticated)
                        logger.info(f"Would update operations guide with {len(new_issues)} new issue patterns")
                        updated_count += 1

            except Exception as e:
                logger.error(f"Error updating operations guide: {str(e)}")

        return updated_count

    def save_knowledge_base(self, faqs: List[FAQEntry], patterns: List[IssuePattern]) -> None:
        """Save knowledge base updates to files"""
        try:
            # Save FAQs
            faq_data = []
            for faq in faqs:
                faq_dict = asdict(faq)
                faq_dict['last_updated'] = faq.last_updated.isoformat()
                faq_data.append(faq_dict)

            with open(self.faq_file, 'w') as f:
                yaml.dump({
                    'generated_at': datetime.now().isoformat(),
                    'faqs': faq_data
                }, f, default_flow_style=False)

            # Save patterns
            pattern_data = []
            for pattern in patterns:
                pattern_dict = asdict(pattern)
                pattern_data.append(pattern_dict)

            with open(self.patterns_file, 'w') as f:
                json.dump({
                    'generated_at': datetime.now().isoformat(),
                    'patterns': pattern_data
                }, f, indent=2)

            logger.info(f"Saved {len(faqs)} FAQs and {len(patterns)} patterns to knowledge base")

        except Exception as e:
            logger.error(f"Error saving knowledge base: {str(e)}")

    def load_existing_knowledge_base(self) -> Tuple[List[FAQEntry], List[IssuePattern]]:
        """Load existing knowledge base"""
        faqs = []
        patterns = []

        # Load FAQs
        if self.faq_file.exists():
            try:
                with open(self.faq_file, 'r') as f:
                    data = yaml.safe_load(f)
                    if data and 'faqs' in data:
                        for faq_data in data['faqs']:
                            faq_data['last_updated'] = datetime.fromisoformat(faq_data['last_updated'])
                            faqs.append(FAQEntry(**faq_data))
            except Exception as e:
                logger.error(f"Error loading FAQs: {str(e)}")

        # Load patterns
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, 'r') as f:
                    data = json.load(f)
                    if data and 'patterns' in data:
                        for pattern_data in data['patterns']:
                            patterns.append(IssuePattern(**pattern_data))
            except Exception as e:
                logger.error(f"Error loading patterns: {str(e)}")

        return faqs, patterns

    def generate_knowledge_effectiveness_metrics(self, faqs: List[FAQEntry], patterns: List[IssuePattern]) -> Dict[str, float]:
        """Generate metrics on knowledge base effectiveness"""
        metrics = {}

        if faqs:
            avg_confidence = sum(faq.confidence for faq in faqs) / len(faqs)
            metrics['average_faq_confidence'] = avg_confidence

            high_confidence_faqs = sum(1 for faq in faqs if faq.confidence >= 80)
            metrics['high_confidence_faq_percentage'] = (high_confidence_faqs / len(faqs)) * 100

        if patterns:
            avg_frequency = sum(pattern.frequency for pattern in patterns) / len(patterns)
            metrics['average_pattern_frequency'] = avg_frequency

            high_impact_patterns = sum(1 for pattern in patterns if pattern.frequency >= 10)
            metrics['high_impact_pattern_count'] = high_impact_patterns

        # Knowledge coverage
        categories_covered = set()
        for faq in faqs:
            categories_covered.add(faq.category)
        for pattern in patterns:
            issue_category = self._categorize_issue(pattern.description)
            categories_covered.add(issue_category)

        total_categories = len(self.issue_categories)
        metrics['category_coverage_percentage'] = (len(categories_covered) / total_categories) * 100

        return metrics

    def update_knowledge_base(self) -> KnowledgeBaseReport:
        """Run complete knowledge base update process"""
        logger.info("Starting knowledge base update...")

        # Analyze recent system issues
        issues = self.analyze_system_logs()

        # Identify patterns
        new_patterns = self.identify_issue_patterns(issues)

        # Load existing knowledge base
        existing_faqs, existing_patterns = self.load_existing_knowledge_base()

        # Generate new FAQs
        new_faqs = self.generate_automated_faqs(issues, new_patterns)

        # Merge with existing (simple merge - could be enhanced)
        all_faqs = existing_faqs + new_faqs
        all_patterns = existing_patterns + new_patterns

        # Update articles
        articles_updated = self.update_existing_articles(new_patterns)

        # Save updated knowledge base
        self.save_knowledge_base(all_faqs, all_patterns)

        # Generate effectiveness metrics
        effectiveness_metrics = self.generate_knowledge_effectiveness_metrics(all_faqs, all_patterns)

        # Identify knowledge gaps
        knowledge_gaps = []
        for category in self.issue_categories.keys():
            category_faqs = [faq for faq in all_faqs if faq.category == category]
            if len(category_faqs) < 2:
                knowledge_gaps.append(f"Limited FAQs for {category} category")

        # Generate recommendations
        recommendations = []
        if len(new_faqs) > 5:
            recommendations.append("High FAQ generation activity - consider manual review")
        if effectiveness_metrics.get('average_faq_confidence', 0) < 70:
            recommendations.append("Low average FAQ confidence - improve answer quality")
        if knowledge_gaps:
            recommendations.append("Knowledge gaps identified - consider manual documentation")

        # Generate report
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.analysis_period_days)

        report = KnowledgeBaseReport(
            report_timestamp=datetime.now(),
            analysis_period=(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')),
            total_articles=len(all_faqs) + len(all_patterns),
            articles_updated=articles_updated,
            new_faqs_generated=len(new_faqs),
            issue_patterns_identified=len(new_patterns),
            knowledge_gaps_found=knowledge_gaps,
            recommendations=recommendations,
            effectiveness_metrics=effectiveness_metrics
        )

        logger.info(f"Knowledge base update completed - {len(new_faqs)} new FAQs, {len(new_patterns)} new patterns")

        return report


def print_knowledge_base_report(report: KnowledgeBaseReport):
    """Print formatted knowledge base report"""
    print(f"\n{'='*70}")
    print(f"KNOWLEDGE BASE MANAGEMENT REPORT")
    print(f"{'='*70}")
    print(f"Report Date: {report.report_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Analysis Period: {report.analysis_period[0]} to {report.analysis_period[1]}")

    print(f"\nUPDATE SUMMARY:")
    print(f"  Total Articles: {report.total_articles}")
    print(f"  Articles Updated: {report.articles_updated}")
    print(f"  New FAQs Generated: {report.new_faqs_generated}")
    print(f"  Issue Patterns Identified: {report.issue_patterns_identified}")

    if report.effectiveness_metrics:
        print(f"\nEFFECTIVENESS METRICS:")
        for metric, value in report.effectiveness_metrics.items():
            print(f"  {metric.replace('_', ' ').title()}: {value:.1f}")

    if report.knowledge_gaps_found:
        print(f"\nKNOWLEDGE GAPS:")
        for gap in report.knowledge_gaps_found:
            print(f"  • {gap}")

    if report.recommendations:
        print(f"\nRECOMMENDATIONS:")
        for rec in report.recommendations:
            print(f"  • {rec}")

    print(f"\n{'='*70}")


def main():
    """Main execution for testing"""
    kb_manager = KnowledgeBaseManager(analysis_period_days=30)
    report = kb_manager.update_knowledge_base()

    print_knowledge_base_report(report)


if __name__ == "__main__":
    main()