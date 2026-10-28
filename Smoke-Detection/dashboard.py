"""Streamlit dashboard for Smoke Detection."""

from __future__ import annotations

import os
import hashlib
import smtplib
from datetime import date, datetime, time, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from backend.authentication import (
    AdministratorAuthenticator,
    AuthenticationConfigurationError,
)
from backend.camera_manager import CameraConfigurationError
from backend.connector_files import ApplicationServices
from backend.email_alert import EmailAlertConfigurationError
from backend.finding_error import configure_error_logging
from backend.pre_checks import PreChecks
from backend.smoke_vehicle_match import SmokeVehicleMatcher
from backend.tracking_id import VehicleTracker
from backend.video_processor import VideoSession


PROJECT_ROOT = Path(__file__).resolve().parent
PAGE_NAMES = (
    "Overview",
    "Live monitoring",
    "Video testing",
    "Detection events",
    "Cameras",
    "Reports",
    "Settings",
)

st.set_page_config(
    page_title="Smoke Detection | Monitoring Console",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_LOGGER = configure_error_logging(PROJECT_ROOT)

st.markdown(
    """
    <style>
      :root { color-scheme: dark; }
      .stApp { background: #0b1220; }
      [data-testid="stSidebar"] { background: #101a2b; border-right: 1px solid #223149; }
      .block-container { padding-top: 2rem; max-width: 1500px; }
      .hero { padding: 1.5rem 1.7rem; border: 1px solid #263750; border-radius: 16px;
              background: linear-gradient(120deg, #14233a, #101a2b); margin-bottom: 1.25rem; }
      .eyebrow { color: #7dd3fc; font-size: .75rem; letter-spacing: .16em; font-weight: 700; }
      .hero h1 { margin: .45rem 0; color: #f2f6fc; font-size: 2rem; }
      .hero p { margin: 0; color: #a8b6c9; font-size: 1rem; }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_services() -> ApplicationServices:
    """Keep one set of shared services across Streamlit reruns."""
    return ApplicationServices(PROJECT_ROOT)


def _utc_bounds(start_day: date | None, end_day: date | None) -> tuple[str | None, str | None]:
    start = (
        datetime.combine(start_day, time.min, tzinfo=timezone.utc).isoformat(timespec="seconds")
        if start_day
        else None
    )
    end = (
        datetime.combine(end_day, time.max, tzinfo=timezone.utc).isoformat(timespec="seconds")
        if end_day
        else None
    )
    return start, end


def _show_login() -> None:
    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">ADMINISTRATOR ACCESS</div>
          <h1>Smoke Detection Console</h1>
          <p>Sign in to manage cameras and review monitoring events.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    try:
        authenticator = AdministratorAuthenticator.from_environment()
    except AuthenticationConfigurationError as exc:
        st.error(str(exc))
        st.info(
            "Set SMOKE_DETECTION_ADMIN_USERNAME and SMOKE_DETECTION_ADMIN_PASSWORD_HASH "
            "in the environment before launching the dashboard. The password hash can be "
            "created with AdministratorAuthenticator.hash_password()."
        )
        return

    with st.form("admin_login"):
        username = st.text_input("Administrator username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
    if submitted:
        result = authenticator.authenticate(username, password)
        if result.authenticated:
            st.session_state["authenticated"] = True
            st.rerun()
        st.error(result.message)


def _overview(services: ApplicationServices) -> None:
    cameras = services.camera_manager.list_cameras()
    event_count = services.event_store.list_events(page=1, page_size=1).total_records
    active_count = sum(camera.enabled for camera in cameras)
    st.subheader("Overview")
    first, second, third = st.columns(3)
    first.metric("Configured cameras", len(cameras))
    second.metric("Enabled cameras", active_count)
    third.metric("Saved CCTV events", event_count)
    if not cameras and event_count == 0:
        st.info("No cameras or detection events are configured yet. Add a camera to begin setup.")
    elif services.monitor is None:
        st.info("Live monitoring is stopped. Start it from Live monitoring when you are ready.")


def _camera_page(services: ApplicationServices) -> None:
    st.subheader("Camera management")
    st.caption("Camera credentials are encrypted in the local database when a protection key is configured.")
    with st.expander("Add a camera", expanded=not services.camera_manager.list_cameras()):
        with st.form("add_camera", clear_on_submit=True):
            display_name = st.text_input("Camera name", placeholder="Northbound gate 1")
            stream_url = st.text_input("Stream address", placeholder="rtsp://camera-address/stream")
            username = st.text_input("Camera username (optional)")
            password = st.text_input("Camera password (optional)", type="password")
            enabled = st.checkbox("Enable this camera", value=True)
            submitted = st.form_submit_button("Save camera", type="primary")
        if submitted:
            try:
                camera = services.camera_manager.add_camera(
                    display_name,
                    stream_url,
                    username=username or None,
                    password=password or None,
                    enabled=enabled,
                )
                st.success(f"Saved {camera.display_name}.")
                st.rerun()
            except (CameraConfigurationError, ValueError) as exc:
                st.error(str(exc))

    cameras = services.camera_manager.list_cameras()
    if not cameras:
        st.info("No cameras have been added.")
        return
    for camera in cameras:
        with st.container(border=True):
            info, controls = st.columns([4, 2])
            with info:
                st.markdown(f"**{camera.display_name}**")
                st.caption(
                    f"Status: {camera.status.replace('_', ' ').title()} · "
                    f"{'Enabled' if camera.enabled else 'Disabled'} · Stream address saved"
                )
            with controls:
                left, right = st.columns(2)
                if left.button("Test stream", key=f"test_{camera.camera_id}", use_container_width=True):
                    try:
                        result = services.camera_manager.test_connection(camera.camera_id)
                        if result.status == "connected":
                            st.success(result.message)
                        else:
                            st.warning(result.message)
                    except CameraConfigurationError as exc:
                        st.error(str(exc))
                if right.button(
                    "Disable" if camera.enabled else "Enable",
                    key=f"toggle_{camera.camera_id}",
                    use_container_width=True,
                ):
                    services.camera_manager.set_enabled(camera.camera_id, not camera.enabled)
                    if services.monitor is not None:
                        services.monitor.refresh()
                    st.rerun()
            with st.expander("Edit camera"):
                st.caption("Leave the credential fields empty to keep the currently saved credentials.")
                with st.form(f"edit_{camera.camera_id}"):
                    edited_name = st.text_input("Camera name", value=camera.display_name)
                    edited_url = st.text_input("Stream address", value=camera.stream_url)
                    replace_credentials = st.checkbox("Replace saved credentials", value=False)
                    edited_username = st.text_input("New camera username", disabled=not replace_credentials)
                    edited_password = st.text_input(
                        "New camera password",
                        type="password",
                        disabled=not replace_credentials,
                    )
                    save_changes = st.form_submit_button("Save changes", type="primary")
                if save_changes:
                    try:
                        services.camera_manager.update_camera(
                            camera.camera_id,
                            display_name=edited_name,
                            stream_url=edited_url,
                            username=edited_username or None,
                            password=edited_password or None,
                            replace_credentials=replace_credentials,
                        )
                        if services.monitor is not None:
                            services.monitor.refresh()
                        st.success("Camera changes were saved.")
                        st.rerun()
                    except (CameraConfigurationError, ValueError) as exc:
                        st.error(str(exc))
                    except KeyError:
                        st.error("The camera no longer exists. Refresh the page and try again.")
                with st.popover("Remove", use_container_width=True):
                    st.warning(f"Remove {camera.display_name} from this project?")
                    if st.button("Confirm removal", key=f"remove_{camera.camera_id}", type="primary"):
                        services.camera_manager.remove_camera(camera.camera_id)
                        if services.monitor is not None:
                            services.monitor.refresh()
                        st.rerun()


def _live_page(services: ApplicationServices) -> None:
    st.subheader("Live monitoring")
    st.caption("Live monitoring uses the configured CCTV streams and the existing project checkpoints.")
    preflight = PreChecks(PROJECT_ROOT, services.runtime_settings, services.camera_manager).run()
    if preflight.ready:
        st.success("System checks passed. Live monitoring can start.")
    else:
        st.error("Live monitoring cannot start until the required checks pass.")
    st.dataframe(
        pd.DataFrame(
            [
                {"Component": check.component, "Status": check.status.title(), "Details": check.message}
                for check in preflight.checks
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    left, right = st.columns(2)
    with left:
        if st.button(
            "Start live monitoring",
            type="primary",
            use_container_width=True,
            disabled=not preflight.ready,
        ):
            try:
                monitor = services.start_monitor()
                monitor.refresh()
                st.success("Monitoring workers have been started for enabled cameras.")
            except Exception as exc:
                st.error(f"Monitoring could not start: {exc}")
    with right:
        if st.button("Stop live monitoring", use_container_width=True):
            services.close()
            st.info("Live monitoring has stopped.")

    if services.monitor is None:
        st.info("Monitoring is stopped. No CCTV stream is being read.")
        return
    views = services.monitor.current_views()
    if not views:
        st.info("There are no cameras configured.")
    for view in views:
        with st.container(border=True):
            st.markdown(f"**{view.camera.display_name}** · {view.status.replace('_', ' ').title()}")
            if view.latest_frame_jpeg:
                st.image(view.latest_frame_jpeg, channels="RGB", use_container_width=True)
            elif view.message:
                st.caption(view.message)
            else:
                st.caption("Waiting for the first frame from this camera.")
    if st.button("Refresh camera views"):
        services.monitor.refresh()
        st.rerun()


def _events_page(services: ApplicationServices) -> None:
    st.subheader("Detection events")
    cameras = services.camera_manager.list_cameras()
    camera_options = {"All cameras": None, **{camera.display_name: camera.camera_id for camera in cameras}}
    selected_camera = st.selectbox("Camera", list(camera_options))
    first_filter, second_filter = st.columns(2)
    with first_filter:
        smoke_options = {
            "All smoke types": None,
            "Black smoke": "black_smoke",
            "Blue smoke": "blue_smoke",
            "White smoke": "white_smoke",
            "Grey smoke": "grey_smoke",
        }
        selected_smoke = st.selectbox("Smoke type", list(smoke_options))
        start_day = st.date_input("Start date (UTC)", value=None, key="event_start")
    with second_filter:
        association_options = {
            "All association states": None,
            "Associated": "associated",
            "Uncertain": "uncertain",
            "Unassociated": "unassociated",
        }
        selected_association = st.selectbox("Vehicle association", list(association_options))
        end_day = st.date_input("End date (UTC)", value=None, key="event_end")
    search = st.text_input("Search camera name, smoke type, or vehicle track ID")
    start_utc, end_utc = _utc_bounds(start_day, end_day)
    result = services.event_store.list_events(
        camera_id=camera_options[selected_camera],
        start_utc=start_utc,
        end_utc=end_utc,
        smoke_classification=smoke_options[selected_smoke],
        association_status=association_options[selected_association],
        search=search or None,
        page=1,
        page_size=500,
    )
    st.caption(f"{result.total_records} matching event(s). Showing up to 500 at a time.")
    if not result.records:
        st.info("No saved CCTV events match these filters.")
        return
    rows = [
        {
            "Time (UTC)": item["timestamp_utc"],
            "Camera": item["camera_name"],
            "Vehicle track ID": item["vehicle_track_id"],
            "Vehicle association": item["association_status"],
            "Smoke type": item["smoke_classification"] or "Unclassified",
            "Classification": item["classification_status"],
            "Plate location": item["plate_status"].replace("_", " ").title(),
            "Plate text": item["license_plate"] or "Unreadable",
            "Detector score": item["smoke_detector_score"],
        }
        for item in result.records
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    csv_data = services.event_store.export_csv(
        camera_id=camera_options[selected_camera],
        start_utc=start_utc,
        end_utc=end_utc,
        smoke_classification=smoke_options[selected_smoke],
        association_status=association_options[selected_association],
        search=search or None,
    )
    st.download_button(
        "Download matching events (CSV)",
        data=csv_data,
        file_name="smoke_detection_events.csv",
        mime="text/csv",
    )


def _video_page(services: ApplicationServices) -> None:
    st.subheader("Video testing")
    st.caption("Uploaded clips are processed in a temporary session. Their results are not saved as CCTV events.")
    upload = st.file_uploader(
        "Choose a video file",
        type=("mp4", "avi", "mov", "mkv"),
        help="Supported formats: MP4, AVI, MOV, and MKV.",
    )
    if upload is None:
        old_session = st.session_state.pop("video_session", None)
        if old_session is not None:
            old_session.close()
        st.session_state.pop("video_result", None)
        st.session_state.pop("video_digest", None)
        st.info("Upload a clip to start a temporary analysis session.")
        return

    video_bytes = upload.getvalue()
    digest = hashlib.sha256(video_bytes).hexdigest()
    if st.session_state.get("video_digest") != digest:
        old_session = st.session_state.get("video_session")
        if old_session is not None:
            old_session.close()
        st.session_state["video_session"] = VideoSession(video_bytes, upload.name)
        st.session_state["video_digest"] = digest
        st.session_state.pop("video_result", None)

    st.caption(f"Selected file: {upload.name} · {len(video_bytes) / (1024 * 1024):.1f} MB")
    if st.button("Process video", type="primary"):
        progress = st.progress(0.0, text="Preparing the existing models…")
        try:
            services.detection_pipeline.load()
            services.color_classifier.load()
            tracker = VehicleTracker(
                services.detection_pipeline.vehicle_model,
                device=services.detection_pipeline.device,
            )

            def update_progress(item: object) -> None:
                value = getattr(item, "progress", None)
                if value is not None:
                    progress.progress(min(1.0, max(0.0, value)), text="Processing video…")

            result = st.session_state["video_session"].process(
                services.detection_pipeline,
                tracker,
                SmokeVehicleMatcher(),
                services.color_classifier,
                services.plate_reader,
                progress_callback=update_progress,
            )
            st.session_state["video_result"] = result
            progress.progress(1.0, text="Video processing complete.")
        except Exception as exc:
            st.error(f"Video processing could not complete: {exc}")

    result = st.session_state.get("video_result")
    if result is None:
        st.info("Choose Process video to run the existing detectors on this clip.")
        return

    metrics = st.columns(3)
    metrics[0].metric("Frames read", result.frames_read)
    metrics[1].metric("Frames analyzed", result.frames_analyzed)
    metrics[2].metric("Smoke events", len(result.events))
    if result.issues:
        for issue in result.issues:
            st.warning(issue)
    st.video(result.output_path)
    if not result.events:
        st.info("No smoke events were returned for this video processing run.")
        return
    rows = [
        {
            "Video time (seconds)": round(event.timestamp_seconds, 2),
            "Smoke type": event.colour.label or "Unclassified",
            "Classification": event.colour.status,
            "Vehicle association": event.association_status,
            "Vehicle track ID": event.vehicle_track_id,
            "Plate location": event.plate.status.replace("_", " ").title(),
            "Plate text": event.plate.text or "Unreadable",
            "Detector score": event.smoke_detector_score,
        }
        for event in result.events
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _reports_page(services: ApplicationServices) -> None:
    st.subheader("Reports")
    cameras = services.camera_manager.list_cameras()
    camera_options = {"All cameras": None, **{camera.display_name: camera.camera_id for camera in cameras}}
    selected_camera = st.selectbox("Report camera", list(camera_options))
    date_range = st.date_input("Report period (UTC)", value=(), key="report_period")
    start_day = date_range[0] if len(date_range) > 0 else None
    end_day = date_range[1] if len(date_range) > 1 else start_day
    start_utc, end_utc = _utc_bounds(start_day, end_day)
    report = services.report_generator.generate(
        start_utc=start_utc,
        end_utc=end_utc,
        camera_id=camera_options[selected_camera],
    )
    st.metric("Matching CCTV events", report.total_events)
    if report.total_events == 0:
        st.info("There are no saved CCTV events in this period.")
        return
    st.caption(f"First event: {report.first_event_utc} · Last event: {report.last_event_utc}")
    one, two, three = st.columns(3)
    with one:
        st.markdown("**Smoke types**")
        st.dataframe(pd.DataFrame(report.smoke_type_counts.items(), columns=["Type", "Events"]), hide_index=True)
    with two:
        st.markdown("**Cameras**")
        st.dataframe(pd.DataFrame(report.camera_counts.items(), columns=["Camera", "Events"]), hide_index=True)
    with three:
        st.markdown("**Vehicle association**")
        st.dataframe(pd.DataFrame(report.association_counts.items(), columns=["Status", "Events"]), hide_index=True)
    st.caption(f"Classification unavailable for {report.classification_unavailable} event(s).")


def _settings_page(services: ApplicationServices) -> None:
    st.subheader("Settings")
    st.markdown("**Project folder**")
    st.code(str(PROJECT_ROOT), language=None)
    st.markdown("**Administrator login**")
    st.write("Set `SMOKE_DETECTION_ADMIN_USERNAME` and `SMOKE_DETECTION_ADMIN_PASSWORD_HASH` before launching.")
    st.markdown("**Camera credential protection**")
    if os.environ.get("SMOKE_DETECTION_CAMERA_KEY"):
        st.success("A camera credential encryption key is configured for this process.")
    else:
        st.warning("SMOKE_DETECTION_CAMERA_KEY is not set. Cameras without saved credentials can still be configured.")
    st.markdown("**Model files**")
    for filename in ("smoke_detector.pt", "vehicle_detector.pt", "plate_detector.pt"):
        model_path = PROJECT_ROOT / "models" / filename
        st.write(f"{'Available' if model_path.is_file() else 'Missing'} · `{filename}`")
    st.markdown("**Active GPU settings**")
    st.write(f"Device: `{services.runtime_settings.device}`")
    st.write(
        "Smoke confidence: "
        f"`{services.runtime_settings.smoke_confidence:.2f}` · "
        "Vehicle confidence: "
        f"`{services.runtime_settings.vehicle_confidence:.2f}` · "
        "Analysis speed: "
        f"`{services.runtime_settings.analysis_fps:g} FPS`"
    )
    st.divider()
    st.markdown("### Email detection alerts")
    st.caption(
        "When enabled, every saved CCTV detection sends this report: camera, UTC time, smoke type, "
        "detector confidence, vehicle association, track ID, and plate result."
    )
    alert_settings = services.email_alerts.get_settings()
    if not os.environ.get("SMOKE_DETECTION_CAMERA_KEY"):
        st.warning("Run setup_security.py before saving an SMTP app password for alerts.")
    with st.form("email_alert_settings"):
        alert_enabled = st.checkbox("Send email for each saved CCTV detection", value=alert_settings.enabled)
        recipients = st.text_input(
            "Alert recipient email address(es)",
            value=", ".join(alert_settings.recipient_addresses),
            placeholder="you@example.com, colleague@example.com",
        )
        smtp_host = st.text_input("SMTP server", value=alert_settings.smtp_host, placeholder="smtp.gmail.com")
        smtp_port = st.number_input("SMTP port", min_value=1, max_value=65535, value=alert_settings.smtp_port)
        use_starttls = st.checkbox("Use STARTTLS encryption", value=alert_settings.use_starttls)
        sender_address = st.text_input("Sender email address", value=alert_settings.sender_address)
        smtp_username = st.text_input("SMTP username", value=alert_settings.username)
        replace_password = st.checkbox("Set or replace SMTP app password", value=False)
        smtp_password = st.text_input(
            "SMTP app password",
            type="password",
            disabled=not replace_password,
            help="For Gmail, create an App Password in your Google Account; do not use your normal email password.",
        )
        saved = st.form_submit_button("Save email alert settings", type="primary")
    if saved:
        try:
            services.email_alerts.save_settings(
                enabled=alert_enabled,
                recipient_addresses=recipients,
                smtp_host=smtp_host,
                smtp_port=int(smtp_port),
                use_starttls=use_starttls,
                sender_address=sender_address,
                username=smtp_username,
                password=smtp_password or None,
                replace_password=replace_password,
            )
            st.success("Email alert settings were saved.")
            st.rerun()
        except EmailAlertConfigurationError as exc:
            st.error(str(exc))
    if alert_settings.password_configured:
        st.caption("An encrypted SMTP app password is saved for this project.")
    if st.button("Send test email", disabled=not alert_settings.password_configured):
        try:
            services.email_alerts.send_test()
            st.success("Test email sent.")
        except (EmailAlertConfigurationError, OSError, smtplib.SMTPException) as exc:
            st.error(f"Test email could not be sent: {type(exc).__name__}.")


if not st.session_state.get("authenticated", False):
    _show_login()
    st.stop()

try:
    app_services = get_services()
except Exception as exc:
    APP_LOGGER.error("Project services could not start: %s", type(exc).__name__)
    st.error(f"Project services could not start: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("### SMOKE DETECTION")
    st.caption("Monitoring and analysis console")
    page = st.radio("Workspace", PAGE_NAMES, label_visibility="collapsed")
    st.divider()
    if st.button("Sign out", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">VEHICLE EMISSIONS MONITORING</div>
      <h1>Smoke Detection Console</h1>
      <p>Review smoke detections from configured CCTV cameras.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

try:
    if page == "Overview":
        _overview(app_services)
    elif page == "Live monitoring":
        _live_page(app_services)
    elif page == "Video testing":
        _video_page(app_services)
    elif page == "Detection events":
        _events_page(app_services)
    elif page == "Cameras":
        _camera_page(app_services)
    elif page == "Reports":
        _reports_page(app_services)
    elif page == "Settings":
        _settings_page(app_services)
except Exception as exc:
    APP_LOGGER.error("Dashboard section failed: page=%s error_type=%s", page, type(exc).__name__)
    st.error(f"This section could not load: {exc}")
