import { Button } from "@/components/ui/button";
import { SessionsPane } from "../../components/sessions/SessionsPane";
import { BellIcon, VolumeIcon } from "./icons";

interface AppShellSidebarProps {
  announcementEnabled: boolean;
  announcementLabel: string;
  notificationLabel: string;
  notificationsEnabled: boolean;
  onBrandClick(): void;
  onNewSession(): void;
  onOpenSettings(): void;
  onLogout(): void;
  onToggleAnnouncements(): void;
  onToggleNotifications(): void;
}

export function AppShellSidebar({
  announcementEnabled,
  announcementLabel,
  notificationLabel,
  notificationsEnabled,
  onBrandClick,
  onNewSession,
  onOpenSettings,
  onLogout,
  onToggleAnnouncements,
  onToggleNotifications,
}: AppShellSidebarProps) {
  return (
    <>
      <header className="sidebarBanner">
        <div className="sidebarBannerActions">
          <Button type="button" variant="ghost" className="brandMark" onClick={onBrandClick}>Codoxear</Button>
          <div className="sidebarActionButtons">
            <Button
              type="button"
              variant="outline"
              size="icon"
              className={`iconAction legacyToggleAction${notificationsEnabled ? " isActive" : ""}`}
              aria-label={notificationLabel}
              title={notificationLabel}
              onClick={onToggleNotifications}
            >
              <BellIcon />
              <span className="visuallyHidden">{notificationLabel}</span>
            </Button>
            <Button
              type="button"
              variant="outline"
              size="icon"
              className={`iconAction legacyToggleAction${announcementEnabled ? " isActive" : ""}`}
              aria-label={announcementLabel}
              title={announcementLabel}
              onClick={onToggleAnnouncements}
            >
              <VolumeIcon />
              <span className="visuallyHidden">{announcementLabel}</span>
            </Button>
          </div>
        </div>
      </header>
      <SessionsPane onNewSession={onNewSession} />
      <footer className="sidebarFooter">
        <Button type="button" variant="outline" className="footerAction"><span className="buttonGlyph">?</span><span>Help</span></Button>
        <Button type="button" variant="outline" className="footerAction" onClick={onOpenSettings}><span className="buttonGlyph">⚙</span><span>Settings</span></Button>
        <Button type="button" variant="outline" className="footerAction" onClick={onLogout}><span className="buttonGlyph">→|</span><span>Log out</span></Button>
      </footer>
    </>
  );
}
