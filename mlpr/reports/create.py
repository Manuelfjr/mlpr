from typing import Any, Dict, Optional
from jinja2 import Environment, FileSystemLoader


class ReportGenerator:
    def __init__(self, template_dir: str) -> None:
        """
        Initialize the ReportGenerator.

        Parameters:
        -----------
            template_dir (str): The directory where the template file is located.
        """
        self.template_dir = template_dir
        
    def render(self, metrics: Dict[str, Any], template_file: str, output_file: str, image_path: Optional[str] = None) -> None:
        """
        Create a report using a template and metrics.

        Parameters:
        -----------
            metrics (Dict[str, Any]): A dictionary containing the metrics.
            template_file (str): The name of the template file.
            output_file (str): The name of the output file.
            image_path (Optional[str]): The path to the image to include in the report.

        Returns:
        --------
            str: A message indicating whether the report was successfully created or not.
        """
        env = Environment(loader=FileSystemLoader(self.template_dir))
        template = env.get_template(template_file)

        report = template.render(metrics=metrics, image_path=image_path)
        try:
            with open(output_file, 'w') as f:
                f.write(report)
            return "Report created successfully."
        except Exception as e:
            return f"Failed to create report. Error: {str(e)}"