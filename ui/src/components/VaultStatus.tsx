import { FolderOpen, RefreshCw, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { syncVault } from "@/lib/backend";



const VaultStatus = () => {
  return (
    <div className="space-y-4">
      <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
        Vault Status
      </h2>
      
      <div className="space-y-3">
        <div className="flex items-center gap-2.5 text-sm">
          <FolderOpen className="h-4 w-4 text-primary" />
          <span className="text-slate-400">Vault path:</span>
          <code className="font-mono text-xs bg-slate-800/50 text-primary px-2 py-0.5 rounded-md border border-slate-700">
            ~/vault/
          </code>
        </div>
        
        <div className="flex items-center gap-2.5 text-sm">
          <FileText className="h-4 w-4 text-primary" />
          <span className="text-slate-400">Files detected:</span>
          <span className="font-semibold text-white">23</span>
        </div>
        
        <div className="flex items-center gap-2.5 text-sm">
          <span className="text-slate-400">Last indexed:</span>
          <span className="text-slate-500">—</span>
        </div>
      </div>
      
      <Button 
        variant="outline" 
        size="sm" 
        className="w-full gap-2 bg-slate-800/50 border-slate-700 text-slate-300 hover:bg-slate-700 hover:text-white hover:border-primary/50"
        onClick={async () => {
          const data = await syncVault();
          console.log("Vault synced:", data);
        }}
      >

        <RefreshCw className="h-3.5 w-3.5" />
        Sync Vault
      </Button>
      
      <p className="text-xs text-slate-500 leading-relaxed">
        Files are read locally. Nothing is uploaded.
      </p>
    </div>
  );
};


export default VaultStatus;
