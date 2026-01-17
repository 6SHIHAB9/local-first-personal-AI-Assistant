import Sidebar from "@/components/Sidebar";
import ChatArea from "@/components/ChatArea";

export default function Index() {
  return (
    <div className="flex min-h-screen w-full">
      <Sidebar />
      <main className="flex-1">
        <ChatArea />
      </main>
    </div>
  );
}
