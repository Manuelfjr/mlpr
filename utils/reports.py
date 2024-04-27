from typing import Any, Dict, Optional
from jinja2 import Environment, FileSystemLoader


def create_report(metrics: Dict[str, Any], template_dir: str, template_file: str, output_file: str, image_path: Optional[str] = None) -> None:
    """
    Create a report using a template and metrics.

    Parameters:
    -----------
        metrics (Dict[str, Any]): A dictionary containing the metrics.
        template_dir (str): The directory where the template file is located.
        template_file (str): The name of the template file.
        output_file (str): The name of the output file.
        image_path (Optional[str]): The path to the image to include in the report.

    Returns:
    --------
        None
    """
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_file)

    report = template.render(metrics=metrics, image_path=image_path)
    with open(output_file, 'w') as f:
        f.write(report)