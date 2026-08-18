"""Behaviour the page must keep, checked against the source.

These are source-level assertions rather than a browser harness. They exist
because both bugs they cover looked like "the button does nothing" from the
user's side, and neither produced a JavaScript error.
"""

from __future__ import annotations

import re

import pytest

from longstay import config

PAGE = config.PROJECT_ROOT / "static" / "index.html"

pytestmark = pytest.mark.skipif(not PAGE.exists(), reason="page missing")


@pytest.fixture(scope="module")
def page() -> str:
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def render_result(page: str) -> str:
    """Just the body of renderResult(), with comments stripped.

    Comments are removed because these tests assert on ORDER, and prose that
    happens to mention an API name would otherwise be mistaken for a call to
    it — which is exactly what happened the first time this was written.
    """
    start = page.index("function renderResult(")
    end = page.index("/* ----------------------------------------------------------- batch */")
    body = page[start:end]
    return re.sub(r"/\*.*?\*/", "", body, flags=re.S)


def test_headline_number_is_written_before_any_animation(render_result):
    """rAF does not run in a background tab. If the count-up is the only path
    that sets the number, the page can display 0.0 while the API returned
    something else — which is worse than showing no number at all.
    """
    assign = render_result.index("big.textContent = target.toFixed(1)")
    first_raf = render_result.index("requestAnimationFrame(")
    assert assign < first_raf, (
        "the final value must be assigned before the first "
        "requestAnimationFrame call"
    )


def test_meter_width_is_set_outside_a_raf_callback(render_result):
    """Same reasoning as the number: the bar must not depend on rAF firing."""
    line = re.search(r"meter\.style\.width = [^;]+;", render_result).group(0)
    before = render_result[: render_result.index(line)]
    assert "requestAnimationFrame(() =>" not in before


def test_result_is_scrolled_into_view_after_rendering(render_result):
    """The sticky form is taller than the viewport on a laptop, so the result
    renders above the fold and the click reads as a no-op."""
    assert "window.scrollTo" in render_result
    assert "getBoundingClientRect" in render_result
    assert "innerHeight" in render_result


def test_scroll_target_clears_the_sticky_topbar(page, render_result):
    """The topbar is 60px and sticky; scrolling to exactly the result top
    would tuck the card under it."""
    offset = re.search(r"scrollY \+ resultBox\.top - (\d+)", render_result)
    assert offset, "no scroll offset found"
    assert int(offset.group(1)) >= 60, "offset does not clear the 60px topbar"


def test_no_duplicate_const_declarations_in_render_result(render_result):
    """A redeclared const is a SyntaxError that kills the ENTIRE script — the
    page then renders with empty dropdowns and no functions defined, and every
    source-level test above still passes. This is the check that would have
    caught it.
    """
    names = re.findall(r"\bconst\s+([A-Za-z_$][\w$]*)\s*=", render_result)
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates, f"const redeclared in the same scope: {duplicates}"


def test_page_script_has_balanced_braces(page):
    """Cheap structural smoke test over the whole script block."""
    script = page[page.index("<script>") + 8: page.rindex("</script>")]
    stripped = re.sub(r"/\*.*?\*/", "", script, flags=re.S)
    stripped = re.sub(r"(?<!:)//[^\n]*", "", stripped)
    for open_c, close_c in (("{", "}"), ("(", ")"), ("[", "]")):
        assert stripped.count(open_c) == stripped.count(close_c), (
            f"unbalanced {open_c}{close_c} in the page script"
        )


def test_motion_is_gated_on_reduced_motion_preference(render_result):
    assert "prefers-reduced-motion" in render_result


def test_the_corrective_scroll_is_instant_not_smooth(render_result):
    """Smooth scrolling is compositor-driven and silently does nothing in some
    contexts. A scroll that does not happen leaves the answer off-screen —
    which is the exact bug this scroll exists to fix.
    """
    assert "window.scrollTo(0, y)" in render_result
    assert "behavior: 'smooth'" not in render_result
    assert "behavior: still" not in render_result


def test_driver_bars_still_appear_when_motion_is_reduced(render_result):
    """Reduced motion must remove the animation, not the information."""
    assert "still ? 0 : 50*i" in render_result


def test_sample_link_scrolls_the_form_into_view(page):
    """The link sits at the bottom of a tall form, but its most visible effect
    — the Cat/Dog switch — is at the top. Without a scroll the click reads as
    doing nothing, which is exactly how it was reported.
    """
    start = page.index("$('#sample').onclick")
    body = page[start: start + 1200]
    assert "window.scrollTo" in body
    assert "getBoundingClientRect" in body
