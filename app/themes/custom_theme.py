from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes


class CustomTheme(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.cyan,
        secondary_hue: colors.Color | str = colors.pink,
        neutral_hue: colors.Color | str = colors.gray,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_none,
        text_size: sizes.Size | str = sizes.text_md,
        font: fonts.Font | str = (
            fonts.GoogleFont("Press Start 2P"),
            "ui-monospace",
            "system-ui",
            "monospace",
        ),
        font_mono: fonts.Font | str = (
            fonts.GoogleFont("Press Start 2P"),
            "ui-monospace",
            "Consolas",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )

        super().set(
            # Darker background for better contrast
            body_background_fill="*neutral_950",
            block_background_fill="*neutral_800",
            block_border_width="1px",
            block_border_color="*primary_400",  # Brighter border for better neon effect
            # Enhanced primary buttons
            button_primary_background_fill="*primary_800",
            button_primary_background_fill_hover="*primary_700",
            button_primary_border_color="*primary_300",  # Brighter border
            button_primary_border_color_hover="*primary_200",
            button_primary_text_color="white",
            button_border_width="1px",
            # Enhanced input fields
            input_background_fill="*neutral_900",
            input_border_color="*primary_400",  # Matches block borders
            input_border_width="1px",
            input_background_fill_dark="*neutral_800",
            # Remove shadows
            shadow_drop="none",
            block_shadow="none",
            # Enhanced secondary buttons
            button_secondary_background_fill="*secondary_800",
            button_secondary_background_fill_hover="*secondary_700",
            button_secondary_text_color="white",
            # Error states with better visibility
            error_background_fill="*neutral_800",
            error_border_color=colors.red.c400,  # Brighter red for better visibility
            # Improve text readability
            body_text_color="*neutral_200",
            body_text_color_subdued="*neutral_400",  # For secondary text
            block_title_text_color="*primary_200",
            block_label_text_color="*neutral_200",
            # Block styling
            block_label_background_fill="*neutral_800",
            block_label_border_color="*primary_500",
            block_label_border_width="1px",
            # Link colors
            link_text_color="*primary_400",
            link_text_color_hover="*primary_300",
            link_text_color_active="*primary_500",
            # Table colors (for any data displays)
            table_border_color="*primary_500",
            table_even_background_fill="*neutral_900",
            table_odd_background_fill="*neutral_800",
            table_row_focus="*primary_900",
            # Form elements
            checkbox_background_color="*neutral_900",
            checkbox_border_color="*primary_400",
            checkbox_label_text_color="*neutral_200",
            checkbox_label_background_fill="*neutral_800",
            checkbox_label_background_fill_dark="*neutral_800",
            checkbox_label_background_fill_hover="*neutral_700",
            checkbox_label_background_fill_hover_dark="*neutral_700",
            # Tab styling - using valid Gradio properties
            background_fill_secondary="*neutral_800",  # Tab background
            background_fill_secondary_dark="*neutral_700",  # Tab hover
            # Add these new styles for inline code
            code_background_fill="*neutral_900",
            # Dropdown styling
            background_fill_primary="*neutral_700",  # Main dropdown background
            border_color_primary="*primary_400",  # Dropdown border
        )
