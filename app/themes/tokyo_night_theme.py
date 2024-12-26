from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes


class TokyoNightTheme(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.indigo,
        secondary_hue: colors.Color | str = colors.purple,
        neutral_hue: colors.Color | str = colors.slate,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_md,
        font: fonts.Font | str = (
            fonts.GoogleFont("JetBrains Mono"),
            "ui-monospace",
            "system-ui",
            "sans-serif",
        ),
        font_mono: fonts.Font | str = (
            fonts.GoogleFont("JetBrains Mono"),
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
            # Main backgrounds
            body_background_fill="#1a1b26",  # TokyoNight deep background
            block_background_fill="#24283b",  # TokyoNight lighter background
            block_border_width="1px",
            block_border_color="#414868",  # Subtle border color
            # Buttons
            button_primary_background_fill="#7aa2f7",  # Light blue
            button_primary_background_fill_hover="#89b4fa",
            button_primary_border_color="#7aa2f7",
            button_primary_border_color_hover="#89b4fa",
            button_primary_text_color="white",
            button_secondary_background_fill="#bb9af7",  # Purple
            button_secondary_background_fill_hover="#c8a6f7",
            button_secondary_text_color="white",
            button_border_width="1px",
            # Input fields
            input_background_fill="#1a1b26",
            input_border_color="#414868",
            input_border_width="1px",
            input_background_fill_dark="#16161e",
            # Text colors
            body_text_color="#c0caf5",  # Light blue-white
            body_text_color_subdued="#565f89",  # Muted color
            block_title_text_color="#7aa2f7",  # Light blue
            block_label_text_color="#c0caf5",
            # Links
            link_text_color="#7dcfff",  # Cyan
            link_text_color_hover="#73daca",  # Mint
            link_text_color_active="#bb9af7",  # Purple
            # Shadows
            shadow_drop="0px 2px 4px rgba(0,0,0,0.2)",
            block_shadow="0px 2px 4px rgba(0,0,0,0.1)",
            # Tables
            table_border_color="#414868",
            table_even_background_fill="#24283b",
            table_odd_background_fill="#1a1b26",
            table_row_focus="#2f334d",
            # Form elements
            checkbox_background_color="#1a1b26",
            checkbox_border_color="#414868",
            checkbox_label_text_color="#c0caf5",
            checkbox_label_background_fill="#24283b",
            checkbox_label_background_fill_hover="#2f334d",
            # Tab styling
            background_fill_secondary="#24283b",
            background_fill_secondary_dark="#16161e",
            # Code blocks
            code_background_fill="#1a1b26",
            # Dropdown
            background_fill_primary="#24283b",
            border_color_primary="#414868",
        )
